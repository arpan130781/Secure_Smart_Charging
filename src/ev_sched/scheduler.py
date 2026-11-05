"""Model Predictive Control (MPC) and CVXPY optimal EV charging scheduler.

Mathematical Formulation:
--------------------------
Let T be the number of discrete time intervals (e.g. 96 slots of dt = 15 minutes = 0.25 hours).
Let N be the number of EV charging sessions i in {0, ..., N-1}.

Given:
- Feeder transformer capacity limit: P_feeder_max in kilowatts [kW].
- Non-EV background base load profile: P_base[t] for t in {0, ..., T-1} in kilowatts [kW].
- Time-of-Use electricity tariff: c[t] for t in {0, ..., T-1} in dollars per kilowatt-hour [$/kWh].
- For each EV session i:
  - Arrival slot: t_arr_i in {0, ..., T-1} [step index]
  - Departure slot: t_dep_i in {1, ..., T} [step index]
  - Energy requested: E_req_i in kilowatt-hours [kWh]
  - Maximum charging power: P_max_i in kilowatts [kW]
- Penalty cost weight for unmet energy: w_unmet in dollars per kilowatt-hour [$/kWh]
  (chosen sufficiently high, e.g. 100.0 $/kWh, so fulfilling requested energy
  strictly dominates energy cost minimization).

Decision Variables:
-------------------
1. P[i, t] >= 0 : Commanded active charging power for vehicle i at slot t in kilowatts [kW].
2. s[i] >= 0    : Penalty slack variable representing unfulfilled energy for vehicle i
                  at departure in kilowatt-hours [kWh].

Objective Function:
-------------------
Minimize total electricity procurement cost plus penalty on unmet energy:
    min  sum_{t=0}^{T-1} c[t] * ( sum_{i=0}^{N-1} P[i, t] ) * dt_hours
         + sum_{i=0}^{N-1} w_unmet * s[i]

Constraints:
------------
1. Per-EV Charging Rate & Plugged-in Window:
   0 <= P[i, t] <= P_max_i   for all t in [t_arr_i, t_dep_i)
   P[i, t] == 0              for all t outside [t_arr_i, t_dep_i)

2. Energy Delivery by Departure (Soft Constraint):
   sum_{t=t_arr_i}^{t_dep_i - 1} P[i, t] * dt_hours + s[i] >= E_req_i
   s[i] >= 0

3. Battery Energy Upper Bound (No Overcharging):
   sum_{t=t_arr_i}^{t_dep_i - 1} P[i, t] * dt_hours <= E_req_i

4. Transformer Capacity Limit (Feeder Security):
   P_base[t] + sum_{i=0}^{N-1} P[i, t] <= P_feeder_max   for all t in {0, ..., T-1}
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import cvxpy as cp
import numpy as np

from ev_sched.hal import EVSEInterface, MeterInterface
from ev_sched.models import EVSession, Feeder, Tariff


@dataclass
class OptimizationResult:
    """Diagnostic outcome of the CVXPY schedule optimization.

    Parameters
    ----------
    status : str
        Solver termination status (e.g., 'optimal', 'infeasible', 'fallback').
    is_success : bool
        Whether a valid optimal or acceptable schedule was obtained.
    power_schedule_kw : np.ndarray
        Matrix of active charging powers of shape (N, T) in kilowatts [kW].
    unmet_energy_kwh : np.ndarray
        Array of slack variables representing unmet energy per session [kWh].
    cost_usd : float
        Calculated electricity cost under the schedule in dollars [$].
    """

    status: str
    is_success: bool
    power_schedule_kw: np.ndarray
    unmet_energy_kwh: np.ndarray
    cost_usd: float


def solve_charging_schedule(
    sessions: List[EVSession],
    feeder: Feeder,
    tariff: Tariff,
    time_step_min: int = 15,
    num_steps: int = 96,
    unmet_penalty_per_kwh: float = 100.0,
    solver: Optional[str] = None,
) -> OptimizationResult:
    """Solve the multi-vehicle optimal charging schedule using CVXPY.

    Parameters
    ----------
    sessions : List[EVSession]
        List of EV charging sessions.
    feeder : Feeder
        Feeder parameters including transformer limit [kW] and base load [kW].
    tariff : Tariff
        Electricity price schedule per 15-minute slot [$/kWh].
    time_step_min : int, default 15
        Duration of each slot in minutes [minutes].
    num_steps : int, default 96
        Number of time steps over the planning horizon.
    unmet_penalty_per_kwh : float, default 100.0
        Cost penalty per unfulfilled kilowatt-hour [$/kWh].
    solver : Optional[str]
        Specific solver name (e.g., 'CLARABEL', 'OSQP', 'SCIPY', 'HIGHS').
        If None, CVXPY selects an available installed solver.

    Returns
    -------
    OptimizationResult
        Complete optimization results including status, power schedule matrix,
        unmet energies, and total electricity cost.
    """
    num_evs = len(sessions)
    dt_hours = time_step_min / 60.0

    if num_evs == 0:
        return OptimizationResult(
            status="optimal",
            is_success=True,
            power_schedule_kw=np.zeros((0, num_steps)),
            unmet_energy_kwh=np.zeros(0),
            cost_usd=0.0,
        )

    # Align base load and tariff profiles
    base_load = list(feeder.base_load_profile_kw)
    if len(base_load) < num_steps:
        pad = base_load[-1] if base_load else 30.0
        base_load.extend([pad] * (num_steps - len(base_load)))
    base_load_arr = np.array(base_load[:num_steps], dtype=float)

    prices = list(tariff.price_per_kwh_per_15min)
    if len(prices) < num_steps:
        pad_price = prices[-1] if prices else 0.15
        prices.extend([pad_price] * (num_steps - len(prices)))
    price_arr = np.array(prices[:num_steps], dtype=float)

    # Decision variables
    # P: shape (num_evs, num_steps), charging power [kW]
    # s: shape (num_evs,), unmet energy slack [kWh]
    P = cp.Variable((num_evs, num_steps), nonneg=True, name="charging_power_kw")
    s = cp.Variable(num_evs, nonneg=True, name="unmet_energy_kwh")

    constraints: List[cp.Constraint] = []

    # 1. Per-EV power saturation and connection windows
    for i, session in enumerate(sessions):
        arr_step = max(0, min(num_steps, session.arrival_min // time_step_min))
        # Quantize departure step
        dep_step = max(arr_step + 1, min(num_steps, int(np.ceil(session.departure_min / time_step_min))))

        # Power upper bound when connected
        constraints.append(P[i, arr_step:dep_step] <= session.max_rate_kw)

        # Zero power when vehicle is not connected
        if arr_step > 0:
            constraints.append(P[i, :arr_step] == 0.0)
        if dep_step < num_steps:
            constraints.append(P[i, dep_step:] == 0.0)

        # 2. Energy requested delivered by departure (soft constraint)
        req_energy = session.remaining_energy_req_kwh
        energy_delivered = cp.sum(P[i, arr_step:dep_step]) * dt_hours
        constraints.append(energy_delivered + s[i] >= req_energy)

        # 3. Do not overcharge beyond requested energy
        constraints.append(energy_delivered <= req_energy)

    # 4. Total feeder load constraint: P_base[t] + sum_i P[i, t] <= transformer_limit_kw
    # Vectorized across all t:
    total_ev_power = cp.sum(P, axis=0)  # shape (num_steps,)
    available_headroom = np.maximum(0.0, feeder.transformer_limit_kw - base_load_arr)
    constraints.append(total_ev_power <= available_headroom)

    # Objective: Minimize electricity purchase cost + unmet energy penalty
    # Total energy cost: sum_t price[t] * total_ev_power[t] * dt_hours
    energy_cost = cp.sum(cp.multiply(price_arr * dt_hours, total_ev_power))
    penalty_cost = unmet_penalty_per_kwh * cp.sum(s)
    objective = cp.Minimize(energy_cost + penalty_cost)

    problem = cp.Problem(objective, constraints)

    # Solve optimization problem
    try:
        # Prefer CLARABEL or OSQP, or fall back to default installed solver
        solvers_to_try = [solver] if solver else [cp.CLARABEL, cp.OSQP, cp.SCIPY]
        solved = False
        last_status = "unsolved"

        for s_name in solvers_to_try:
            if s_name is None:
                continue
            if s_name in cp.installed_solvers():
                try:
                    problem.solve(solver=s_name)
                    last_status = problem.status
                    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                        solved = True
                        break
                except Exception:
                    continue

        if not solved:
            # Fall back to default solver without specifying
            problem.solve()
            last_status = problem.status
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                solved = True

        if solved and P.value is not None:
            power_matrix = np.maximum(0.0, np.array(P.value, dtype=float))
            # Filter numerical solver residual noise below 1e-5 kW (0.01 Watt)
            power_matrix = np.where(power_matrix < 1e-5, 0.0, power_matrix)
            unmet_arr = (
                np.maximum(0.0, np.array(s.value, dtype=float))
                if s.value is not None
                else np.zeros(num_evs)
            )
            unmet_arr = np.where(unmet_arr < 1e-5, 0.0, unmet_arr)
            calculated_cost = float(np.sum(price_arr * dt_hours * np.sum(power_matrix, axis=0)))
            return OptimizationResult(
                status=str(problem.status),
                is_success=True,
                power_schedule_kw=power_matrix,
                unmet_energy_kwh=unmet_arr,
                cost_usd=calculated_cost,
            )
        else:
            # Infeasible or failed: safe fallback
            return _safe_fallback_result(
                sessions=sessions,
                base_load_arr=base_load_arr,
                transformer_limit_kw=feeder.transformer_limit_kw,
                price_arr=price_arr,
                dt_hours=dt_hours,
                num_steps=num_steps,
                status=str(last_status),
            )

    except Exception as exc:
        # Robust fallback: never crash on solver exception
        return _safe_fallback_result(
            sessions=sessions,
            base_load_arr=base_load_arr,
            transformer_limit_kw=feeder.transformer_limit_kw,
            price_arr=price_arr,
            dt_hours=dt_hours,
            num_steps=num_steps,
            status=f"error_{type(exc).__name__}",
        )


def _safe_fallback_result(
    sessions: List[EVSession],
    base_load_arr: np.ndarray,
    transformer_limit_kw: float,
    price_arr: np.ndarray,
    dt_hours: float,
    num_steps: int,
    status: str,
) -> OptimizationResult:
    """Generate safe, conservative fallback schedule without violating transformer limit.

    When the optimization is infeasible (e.g. background load already exceeds limit),
    safely reduce power setpoints to zero (minimum rate) to protect grid equipment.
    """
    num_evs = len(sessions)
    fallback_power = np.zeros((num_evs, num_steps), dtype=float)
    unmet_energy = np.array([s.remaining_energy_req_kwh for s in sessions], dtype=float)

    return OptimizationResult(
        status=status,
        is_success=False,
        power_schedule_kw=fallback_power,
        unmet_energy_kwh=unmet_energy,
        cost_usd=0.0,
    )


@dataclass
class MPCSimulationResult:
    """Comprehensive diagnostic outputs from the receding-horizon MPC loop.

    Parameters
    ----------
    step_times_min : List[int]
        Elapsed simulation time at each 15-minute interval [minutes].
    step_times_hours : List[float]
        Elapsed simulation time at each interval [hours].
    measured_feeder_kw : List[float]
        [MEASURED] Feeder active power telemetry read via MeterInterface [kW].
    total_ev_power_kw : List[float]
        [MEASURED & SIMULATED] Aggregate dispatched EV active power [kW].
    base_load_kw : List[float]
        [SIMULATED] Non-EV background grid active power [kW].
    transformer_limit_kw : float
        Substation transformer capacity rating [kW].
    overload_kw : List[float]
        Active power overload exceeding transformer limit [kW].
    solver_statuses : List[str]
        Recorded solver status at each re-optimization step.
    sessions : List[EVSession]
        EV sessions with final delivered energies.
    """

    step_times_min: List[int]
    step_times_hours: List[float]
    measured_feeder_kw: List[float]
    total_ev_power_kw: List[float]
    base_load_kw: List[float]
    transformer_limit_kw: float
    overload_kw: List[float]
    solver_statuses: List[str]
    sessions: List[EVSession]


def run_mpc_charging_loop(
    feeder: Feeder,
    sessions: List[EVSession],
    tariff: Tariff,
    meter: MeterInterface,
    evse: EVSEInterface,
    simulator: Any,
    time_step_min: int = 15,
    num_steps: int = 96,
    unmet_penalty_per_kwh: float = 100.0,
    horizon_steps: Optional[int] = None,
) -> MPCSimulationResult:
    """Execute receding-horizon MPC EV charging simulation.

    At each discrete 15-minute time step:
    1. Reads the feeder meter through MeterInterface.
    2. Re-solves the constrained CVXPY optimization over the remaining horizon.
    3. Dispatches only the first step's optimal power setpoints via EVSEInterface.
    4. Advances physical simulation state by one time step.

    Parameters
    ----------
    feeder : Feeder
        Feeder parameters including transformer limit [kW] and base load [kW].
    sessions : List[EVSession]
        Fleet of EV charging sessions.
    tariff : Tariff
        Time-of-Use electricity price schedule [$/kWh].
    meter : MeterInterface
        HAL interface providing feeder power telemetry.
    evse : EVSEInterface
        HAL interface for dispatching EV charging setpoints.
    simulator : Any
        Physical simulator to step forward.
    time_step_min : int, default 15
        Step duration in minutes [minutes].
    num_steps : int, default 96
        Total time steps in 24-hour simulation.
    unmet_penalty_per_kwh : float, default 100.0
        Cost weight on unfulfilled target energy [$/kWh].
    horizon_steps : Optional[int]
        Maximum lookahead planning horizon in steps. If None, plans over all
        remaining steps until the end of the 24-hour day.

    Returns
    -------
    MPCSimulationResult
        Trajectory history including feeder loads, overloads, and solver statuses.
    """
    dt_hours = time_step_min / 60.0

    step_times_min: List[int] = []
    step_times_hours: List[float] = []
    measured_feeder_kw: List[float] = []
    total_ev_power_kw: List[float] = []
    base_load_kw: List[float] = []
    overload_kw: List[float] = []
    solver_statuses: List[str] = []

    # Reset delivered energy on all sessions
    for s in sessions:
        s.delivered_energy_kwh = 0.0

    base_profile = list(feeder.base_load_profile_kw)
    if len(base_profile) < num_steps:
        pad = base_profile[-1] if base_profile else 30.0
        base_profile.extend([pad] * (num_steps - len(base_profile)))

    price_profile = list(tariff.price_per_kwh_per_15min)
    if len(price_profile) < num_steps:
        pad_p = price_profile[-1] if price_profile else 0.15
        price_profile.extend([pad_p] * (num_steps - len(price_profile)))

    for step in range(num_steps):
        t_min = step * time_step_min
        step_times_min.append(t_min)
        step_times_hours.append(t_min / 60.0)

        # 1. Read feeder meter via HAL
        meas_kw = meter.read_feeder_kw(t_min)
        measured_feeder_kw.append(meas_kw)

        curr_base_kw = base_profile[step]
        base_load_kw.append(curr_base_kw)

        # 2. Determine remaining horizon for re-solving
        rem_steps = num_steps - step
        plan_steps = rem_steps if horizon_steps is None else min(horizon_steps, rem_steps)

        # Prepare remaining profiles
        rem_base = base_profile[step : step + plan_steps]
        rem_feeder = Feeder(
            transformer_limit_kw=feeder.transformer_limit_kw,
            base_load_profile_kw=rem_base,
        )
        rem_prices = price_profile[step : step + plan_steps]
        rem_tariff = Tariff(price_per_kwh_per_15min=rem_prices)

        # Construct time-shifted sessions relative to current time t_min
        active_rem_sessions: List[EVSession] = []
        for s in sessions:
            # Shift arrival and departure relative to current time step
            arr_rel = max(0, s.arrival_min - t_min)
            dep_rel = max(1, s.departure_min - t_min)

            # Check if vehicle has arrived and not yet departed
            if s.is_connected(t_min) or (s.arrival_min >= t_min and s.arrival_min < t_min + plan_steps * time_step_min):
                rem_energy = s.remaining_energy_req_kwh
            else:
                rem_energy = 0.0

            shifted_s = EVSession(
                arrival_min=arr_rel,
                departure_min=dep_rel,
                energy_req_kwh=rem_energy,
                max_rate_kw=s.max_rate_kw,
                session_id=s.session_id,
                delivered_energy_kwh=0.0,
            )
            active_rem_sessions.append(shifted_s)

        # 3. Solve CVXPY schedule over the remaining horizon
        opt_res = solve_charging_schedule(
            sessions=active_rem_sessions,
            feeder=rem_feeder,
            tariff=rem_tariff,
            time_step_min=time_step_min,
            num_steps=plan_steps,
            unmet_penalty_per_kwh=unmet_penalty_per_kwh,
        )
        solver_statuses.append(opt_res.status)

        # 4. Extract and apply ONLY the first step's setpoints
        step_ev_power = 0.0
        for i, s in enumerate(sessions):
            if opt_res.is_success and i < opt_res.power_schedule_kw.shape[0]:
                cmd_kw = float(opt_res.power_schedule_kw[i, 0])
            else:
                # Safe fallback if solver failed: zero or safe minimal rate
                cmd_kw = 0.0

            # Dispatch via EVSEInterface
            evse.set_setpoint_kw(s.session_id, cmd_kw)
            actual_kw = evse.read_session_kw(s.session_id)
            step_ev_power += actual_kw

        total_ev_power_kw.append(step_ev_power)
        actual_total_feeder = curr_base_kw + step_ev_power
        overload = max(0.0, actual_total_feeder - feeder.transformer_limit_kw)
        overload_kw.append(overload)

        # 5. Advance simulator by one time step
        simulator.step()

    return MPCSimulationResult(
        step_times_min=step_times_min,
        step_times_hours=step_times_hours,
        measured_feeder_kw=measured_feeder_kw,
        total_ev_power_kw=total_ev_power_kw,
        base_load_kw=base_load_kw,
        transformer_limit_kw=feeder.transformer_limit_kw,
        overload_kw=overload_kw,
        solver_statuses=solver_statuses,
        sessions=sessions,
    )
