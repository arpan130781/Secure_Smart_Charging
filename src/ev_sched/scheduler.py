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

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp
import numpy as np

from ev_sched.hal import EVSEInterface, MeterInterface
from ev_sched.models import EVSession, Feeder, Tariff, resolve_session_ids

logger = logging.getLogger(__name__)


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
    solver_used : Optional[str], default None
        Name of the solver that successfully found the solution or was attempted.
    is_inaccurate : bool, default False
        True if the solver status was OPTIMAL_INACCURATE.
    """

    status: str
    is_success: bool
    power_schedule_kw: np.ndarray
    unmet_energy_kwh: np.ndarray
    cost_usd: float
    solver_used: str | None = None
    is_inaccurate: bool = False


def validate_schedule_constraints(
    power_matrix: np.ndarray,
    unmet_arr: np.ndarray,
    sessions: list[EVSession],
    base_load_arr: np.ndarray,
    transformer_limit_kw: float,
    dt_hours: float,
    tol_kw: float = 0.05,
    tol_kwh: float = 0.05,
    max_aggregate_ev_kw: float | None = None,
    num_steps: int | None = None,
) -> tuple[bool, str]:
    """Check whether a candidate power schedule satisfies physical constraints within tolerance.

    Parameters
    ----------
    power_matrix : np.ndarray
        Shape (N, T) matrix of candidate charging powers in kilowatts [kW].
    unmet_arr : np.ndarray
        Shape (N,) vector of unmet energy slack values in kilowatt-hours [kWh].
    sessions : list[EVSession]
        Active EV sessions.
    base_load_arr : np.ndarray
        Shape (T,) array of background feeder base load in kilowatts [kW].
    transformer_limit_kw : float
        Substation transformer capacity limit in kilowatts [kW].
    dt_hours : float
        Time step in hours (e.g. 0.25 h).
    tol_kw : float, default 0.05
        Numerical tolerance for power constraints [kW].
    tol_kwh : float, default 0.05
        Numerical tolerance for energy constraints [kWh].
    max_aggregate_ev_kw : Optional[float], default None
        Optional independent aggregate EV power limit [kW].
    num_steps : Optional[int], default None
        Expected number of time steps (T). If None, defaults to len(base_load_arr).

    Returns
    -------
    tuple[bool, str]
        (is_valid, failure_reason)
    """
    # 1. Validate numerical scalar parameters
    if not np.isfinite(dt_hours) or dt_hours <= 0.0:
        return False, f"Invalid dt_hours: {dt_hours}; must be finite and strictly positive"
    if not np.isfinite(transformer_limit_kw) or transformer_limit_kw <= 0.0:
        return False, f"Invalid transformer_limit_kw: {transformer_limit_kw}; must be finite and strictly positive"
    if not np.isfinite(tol_kw) or tol_kw < 0.0 or not np.isfinite(tol_kwh) or tol_kwh < 0.0:
        return False, "Tolerance parameters (tol_kw, tol_kwh) must be finite and non-negative"

    # 2. Validate base load array
    if not isinstance(base_load_arr, np.ndarray):
        return False, "Base load profile must be a numpy ndarray"
    if base_load_arr.ndim != 1:
        return False, f"Base load profile must be 1D array, got {base_load_arr.ndim}D"
    if not np.all(np.isfinite(base_load_arr)):
        return False, "Non-finite values (NaN/Inf) detected in base load profile"
    if np.any(base_load_arr < -tol_kw):
        return False, "Negative values detected in base load profile"

    expected_num_evs = len(sessions)
    expected_num_steps = num_steps if num_steps is not None else len(base_load_arr)

    if base_load_arr.shape != (expected_num_steps,):
        return False, f"Invalid base load shape {base_load_arr.shape}; expected ({expected_num_steps},)"

    # 3. Validate schedule matrices and vectors
    if not isinstance(power_matrix, np.ndarray) or not isinstance(unmet_arr, np.ndarray):
        return False, "Power schedule and unmet energy must be numpy ndarrays"

    if power_matrix.ndim != 2:
        return False, f"Power schedule must be 2D array, got {power_matrix.ndim}D"
    if power_matrix.shape != (expected_num_evs, expected_num_steps):
        return False, f"Invalid power schedule shape {power_matrix.shape}; expected ({expected_num_evs}, {expected_num_steps})"

    if unmet_arr.ndim != 1:
        return False, f"Unmet energy must be 1D array, got {unmet_arr.ndim}D"
    if unmet_arr.shape != (expected_num_evs,):
        return False, f"Invalid unmet energy array shape {unmet_arr.shape}; expected ({expected_num_evs},)"

    if not np.all(np.isfinite(power_matrix)):
        return False, "Non-finite values (NaN/Inf) detected in power schedule"
    if not np.all(np.isfinite(unmet_arr)):
        return False, "Non-finite values (NaN/Inf) detected in unmet energy vector"

    if np.any(power_matrix < -tol_kw):
        min_p = float(np.min(power_matrix))
        return False, f"Negative charging power detected: {min_p:.3f} kW < -{tol_kw} kW"

    if np.any(unmet_arr < -tol_kwh):
        min_u = float(np.min(unmet_arr))
        return False, f"Negative unmet energy slack detected: {min_u:.3f} kWh < -{tol_kwh} kWh"

    time_step_min = round(dt_hours * 60)
    for i, s in enumerate(sessions):
        p_row = power_matrix[i, :]
        max_rate = s.max_rate_kw

        for t in range(expected_num_steps):
            t_start = t * time_step_min
            t_end = (t + 1) * time_step_min
            frac = s.connection_fraction(t_start, t_end)
            if frac <= 0.0 and p_row[t] > tol_kw:
                return (
                    False,
                    f"Session {s.session_id} scheduled outside connection window at step {t}: {p_row[t]:.3f} kW",
                )
            effective_max = max_rate * frac
            if p_row[t] > effective_max + tol_kw:
                return (
                    False,
                    f"Session {s.session_id} exceeds max rate at step {t}: {p_row[t]:.3f} kW > allowable {effective_max:.3f} kW",
                )

        energy_del = float(np.sum(p_row) * dt_hours)
        if energy_del > s.remaining_energy_req_kwh + tol_kwh:
            return (
                False,
                f"Session {s.session_id} energy delivered {energy_del:.3f} kWh exceeds target {s.remaining_energy_req_kwh:.3f} kWh",
            )

        # Check unmet energy slack fulfillment: energy_del + s[i] >= target_energy
        dwell_horizon_min = s.overlap_minutes(0, expected_num_steps * time_step_min)
        full_dwell_min = max(1.0, float(s.duration_min))
        if s.departure_min > expected_num_steps * time_step_min:
            target_energy = min(
                s.remaining_energy_req_kwh,
                s.remaining_energy_req_kwh * (dwell_horizon_min / full_dwell_min),
            )
        else:
            target_energy = s.remaining_energy_req_kwh

        if energy_del + unmet_arr[i] < target_energy - tol_kwh:
            return (
                False,
                f"Session {s.session_id} unmet energy balance violated: delivered {energy_del:.3f} + slack {unmet_arr[i]:.3f} < target {target_energy:.3f} kWh",
            )

    total_ev = np.sum(power_matrix, axis=0) if expected_num_evs > 0 else np.zeros(expected_num_steps)
    total_feeder = base_load_arr + total_ev
    if np.any(total_feeder > transformer_limit_kw + tol_kw):
        max_feeder = float(np.max(total_feeder))
        return False, f"Transformer capacity violated: peak {max_feeder:.3f} kW > limit {transformer_limit_kw:.3f} kW"

    if max_aggregate_ev_kw is not None and np.any(total_ev > max_aggregate_ev_kw + tol_kw):
        max_ev = float(np.max(total_ev))
        return False, f"Independent aggregate EV limit violated: peak {max_ev:.3f} kW > limit {max_aggregate_ev_kw:.3f} kW"

    return True, "All constraints satisfied within tolerance"


def solve_charging_schedule(
    sessions: list[EVSession],
    feeder: Feeder,
    tariff: Tariff,
    time_step_min: int = 15,
    num_steps: int = 96,
    unmet_penalty_per_kwh: float = 100.0,
    solver: str | None = None,
    max_aggregate_ev_kw: float | None = None,
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
    resolve_session_ids(sessions)
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

    if max_aggregate_ev_kw is not None and max_aggregate_ev_kw <= 0.0:
        return OptimizationResult(
            status="optimal",
            is_success=True,
            power_schedule_kw=np.zeros((num_evs, num_steps)),
            unmet_energy_kwh=np.array([s.energy_req_kwh for s in sessions]),
            cost_usd=0.0,
            solver_used="zero_aggregate_clamp",
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

    constraints: list[cp.Constraint] = []

    # 1. Per-EV power saturation and connection windows
    for i, session in enumerate(sessions):
        effective_max_rates = np.array(
            [
                session.max_rate_kw * session.connection_fraction(t * time_step_min, (t + 1) * time_step_min)
                for t in range(num_steps)
            ],
            dtype=float,
        )

        # Power upper bound per step (accounts for fractional connection in boundary steps)
        constraints.append(P[i, :] <= effective_max_rates)

        # Zero power when vehicle is not connected at all during a step
        zero_steps = np.where(effective_max_rates <= 0.0)[0]
        if len(zero_steps) > 0:
            constraints.append(P[i, zero_steps] == 0.0)

        # 2. Energy requested delivered by departure (soft constraint)
        req_energy = session.remaining_energy_req_kwh
        energy_delivered = cp.sum(P[i, :]) * dt_hours

        # When the vehicle's dwell extends beyond the receding horizon (departure_min > num_steps * time_step_min),
        # prorate the horizon target energy by the fraction of dwell within this horizon to prevent
        # artificial peak charging and false unmet penalties.
        dwell_horizon_min = session.overlap_minutes(0, num_steps * time_step_min)
        full_dwell_min = max(1.0, float(session.duration_min))
        if session.departure_min > num_steps * time_step_min:
            target_energy = min(req_energy, req_energy * (dwell_horizon_min / full_dwell_min))
        else:
            target_energy = req_energy

        constraints.append(energy_delivered + s[i] >= target_energy)

        # 3. Do not overcharge beyond requested energy
        constraints.append(energy_delivered <= req_energy)

    # 4. Total feeder load constraint: P_base[t] + sum_i P[i, t] <= transformer_limit_kw
    # Vectorized across all t:
    total_ev_power = cp.sum(P, axis=0)  # shape (num_steps,)
    available_headroom = feeder.transformer_limit_kw - base_load_arr
    constraints.append(total_ev_power <= available_headroom)

    # 5. Independent Protective Constraint (if configured):
    # Defense-in-depth: enforces an independent ceiling on aggregate EV charging power
    # decoupled from potentially spoofed feeder meter telemetry or base load forecast.
    if max_aggregate_ev_kw is not None:
        constraints.append(total_ev_power <= max_aggregate_ev_kw)

    # Objective: Minimize electricity purchase cost + unmet energy penalty
    # Total energy cost: sum_t price[t] * total_ev_power[t] * dt_hours
    energy_cost = cp.sum(cp.multiply(price_arr * dt_hours, total_ev_power))
    penalty_cost = unmet_penalty_per_kwh * cp.sum(s)
    objective = cp.Minimize(energy_cost + penalty_cost)

    problem = cp.Problem(objective, constraints)

    # Solve optimization problem
    if solver is not None:
        solvers_to_try = [solver]
    else:
        candidates = [cp.CLARABEL, cp.OSQP, cp.SCIPY]
        solvers_to_try = [s_name for s_name in candidates if s_name in cp.installed_solvers()]
        if not solvers_to_try:
            solvers_to_try = cp.installed_solvers()

    solved = False
    successful_solver: str | None = None
    is_inaccurate = False
    last_status = "unsolved"
    last_solver: str | None = None

    for s_name in solvers_to_try:
        last_solver = s_name
        try:
            problem.solve(solver=s_name)
            last_status = str(problem.status)
        except cp.error.SolverError as exc:
            logger.warning("Solver '%s' failed with SolverError: %s", s_name, exc)
            last_status = "solver_error"
            if solver is not None:
                # If caller passed an explicit solver, never substitute another silently
                break
            continue

        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            is_opt_inaccurate = (problem.status == cp.OPTIMAL_INACCURATE)
            if is_opt_inaccurate:
                logger.warning("Solver '%s' returned non-optimal status: %s", s_name, problem.status)

            if P.value is not None:
                raw_p = np.array(P.value, dtype=float)
                raw_u = (
                    np.array(s.value, dtype=float)
                    if s.value is not None
                    else np.zeros(num_evs)
                )

                # Step 1: Validate RAW solver output BEFORE any clipping or preprocessing
                is_raw_valid, raw_reason = validate_schedule_constraints(
                    power_matrix=raw_p,
                    unmet_arr=raw_u,
                    sessions=sessions,
                    base_load_arr=base_load_arr,
                    transformer_limit_kw=feeder.transformer_limit_kw,
                    dt_hours=dt_hours,
                    tol_kw=0.05,
                    tol_kwh=0.05,
                    max_aggregate_ev_kw=max_aggregate_ev_kw,
                    num_steps=num_steps,
                )
                if not is_raw_valid:
                    logger.error(
                        "Solver '%s' status '%s' raw solution failed constraint validation: %s. "
                        "Rejecting raw solution without clipping.",
                        s_name,
                        problem.status,
                        raw_reason,
                    )
                    last_status = f"{problem.status.lower()}_invalid"
                    is_inaccurate = is_opt_inaccurate
                    if solver is not None:
                        break
                    continue

                # Step 2: Form final candidate dispatch schedule (zeroing small numerical noise < 1e-5)
                cand_p = np.maximum(0.0, raw_p)
                cand_p = np.where(cand_p < 1e-5, 0.0, cand_p)
                cand_u = np.maximum(0.0, raw_u)
                cand_u = np.where(cand_u < 1e-5, 0.0, cand_u)

                # Step 3: Validate the final schedule that will actually be dispatched
                is_final_valid, final_reason = validate_schedule_constraints(
                    power_matrix=cand_p,
                    unmet_arr=cand_u,
                    sessions=sessions,
                    base_load_arr=base_load_arr,
                    transformer_limit_kw=feeder.transformer_limit_kw,
                    dt_hours=dt_hours,
                    tol_kw=0.05,
                    tol_kwh=0.05,
                    max_aggregate_ev_kw=max_aggregate_ev_kw,
                    num_steps=num_steps,
                )
                if is_final_valid:
                    solved = True
                    successful_solver = s_name
                    is_inaccurate = is_opt_inaccurate
                    break
                else:
                    logger.error(
                        "Solver '%s' status '%s' final dispatch schedule failed constraint validation: %s. "
                        "Rejecting candidate solution.",
                        s_name,
                        problem.status,
                        final_reason,
                    )
                    last_status = f"{problem.status.lower()}_invalid"
                    is_inaccurate = is_opt_inaccurate
                    if solver is not None:
                        break
                    continue
            else:
                last_status = f"{problem.status.lower()}_invalid"
                is_inaccurate = is_opt_inaccurate
                if solver is not None:
                    break
                continue
        elif problem.status in [
            cp.INFEASIBLE,
            cp.INFEASIBLE_INACCURATE,
            cp.UNBOUNDED,
            cp.UNBOUNDED_INACCURATE,
        ]:
            logger.warning(
                "Problem formulation is %s under solver '%s'; stopping solver search.",
                problem.status,
                s_name,
            )
            break
        else:
            logger.warning(
                "Solver '%s' returned non-optimal status: %s",
                s_name,
                problem.status,
            )
            if solver is not None:
                break

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
            solver_used=successful_solver,
            is_inaccurate=is_inaccurate,
        )
    else:
        # Infeasible, unbounded, or failed: safe fallback
        return _safe_fallback_result(
            sessions=sessions,
            base_load_arr=base_load_arr,
            transformer_limit_kw=feeder.transformer_limit_kw,
            price_arr=price_arr,
            dt_hours=dt_hours,
            num_steps=num_steps,
            status=str(last_status),
            solver_used=last_solver,
            is_inaccurate=is_inaccurate,
        )


def _safe_fallback_result(
    sessions: list[EVSession],
    base_load_arr: np.ndarray,
    transformer_limit_kw: float,
    price_arr: np.ndarray,
    dt_hours: float,
    num_steps: int,
    status: str,
    solver_used: str | None = None,
    is_inaccurate: bool = False,
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
        solver_used=solver_used,
        is_inaccurate=is_inaccurate,
    )


def solve_water_filling_step(
    transformer_limit_kw: float,
    base_load_kw: float,
    sessions: list[EVSession],
    current_time_min: int,
    dt_hours: float = 0.25,
    safety_margin_kw: float = 0.0,
    fail_safe_active: bool = False,
    safe_cap_kw: float = 0.0,
) -> dict[str, float]:
    """Execute deterministic priority water-filling dispatch for the current time step.

    Runs in O(N) time with zero dynamic memory allocation. Bounded strictly by physical
    feeder headroom, individual EV maximum charging rates, and unmet energy requests.
    Attempts invocation via the ANSI C core (core_c/build/libseccharge) if compiled,
    falling back to pure Python arithmetic.
    """
    resolved = resolve_session_ids(sessions)
    if not sessions:
        return {}
    s_ids = [s.session_id for s in resolved]

    # Extract active sessions plugged in at current_time_min
    session_data = []
    for s in sessions:
        is_plugged = s.arrival_min <= current_time_min < s.departure_min
        rem_energy = max(0.0, s.energy_req_kwh - s.delivered_energy_kwh)
        dwell_h = max(0.0, (s.departure_min - current_time_min) / 60.0)
        session_data.append({
            "max_rate_kw": s.max_rate_kw if is_plugged else 0.0,
            "energy_requested_kwh": s.energy_req_kwh,
            "energy_delivered_kwh": s.delivered_energy_kwh,
            "remaining_energy_kwh": rem_energy if is_plugged else 0.0,
            "dwell_hours": dwell_h if is_plugged else 0.0,
        })

    # Try C core invocation
    try:
        from ev_sched.c_bridge import LIB_AVAILABLE, c_schedule_water_filling

        if LIB_AVAILABLE:
            _, allocations = c_schedule_water_filling(
                transformer_limit_kw=transformer_limit_kw,
                base_load_kw=base_load_kw,
                safety_margin_kw=safety_margin_kw,
                time_step_hours=dt_hours,
                sessions_data=session_data,
                fail_safe_active=fail_safe_active,
                safe_cap_kw=safe_cap_kw,
            )
            return {s_ids[i]: allocations[i] for i in range(len(sessions))}
    except (ImportError, RuntimeError, OSError):
        logger.debug("libseccharge C core not available; using pure Python water-filling fallback.")

    # Pure Python zero-heap-style water-filling fallback
    raw_headroom = transformer_limit_kw - base_load_kw - safety_margin_kw
    headroom = max(0.0, raw_headroom)
    if fail_safe_active:
        headroom = min(headroom, max(0.0, safe_cap_kw))

    if headroom <= 0.0:
        return {sid: 0.0 for sid in s_ids}

    # Caps and weights
    caps = []
    weights = []
    for s_info in session_data:
        rem_e = s_info["remaining_energy_kwh"]
        max_p = s_info["max_rate_kw"]
        dwell = s_info["dwell_hours"]
        if dwell <= 0.0 or rem_e <= 1e-6 or max_p <= 0.0:
            caps.append(0.0)
            weights.append(0.0)
        else:
            e_cap = rem_e / dt_hours
            p_cap = min(max_p, e_cap)
            caps.append(p_cap)
            eff_dwell = max(dwell, dt_hours)
            weights.append(max(0.001, rem_e / eff_dwell))

    if sum(caps) <= headroom:
        return {s_ids[i]: caps[i] for i in range(len(sessions))}

    # Water-filling active-set iteration
    allocations = [0.0] * len(sessions)
    is_frozen = [caps[i] == 0.0 for i in range(len(sessions))]
    rem_h = headroom

    for _ in range(len(sessions)):
        active_w_sum = sum(weights[i] for i in range(len(sessions)) if not is_frozen[i])
        if active_w_sum <= 0.0 or rem_h <= 0.0:
            break
        clipped = False
        for i in range(len(sessions)):
            if is_frozen[i]:
                continue
            trial = (weights[i] / active_w_sum) * rem_h
            if trial >= caps[i]:
                allocations[i] = caps[i]
                rem_h -= caps[i]
                is_frozen[i] = True
                clipped = True
        if not clipped:
            for i in range(len(sessions)):
                if not is_frozen[i]:
                    allocations[i] = (weights[i] / active_w_sum) * rem_h
                    is_frozen[i] = True
            break

    return {s_ids[i]: allocations[i] for i in range(len(sessions))}



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
    true_feeder_kw : List[float]
        [SIMULATED] Ground-truth total active power flowing through substation [kW].
    alarms_raised : List[bool]
        Whether detector flagged an alarm at each step.
    safe_mode_active : List[bool]
        Whether controller operated in safe mode at each step.
    solvers_used : List[Optional[str]]
        Recorded solver used at each step.
    inaccurate_statuses : List[bool]
        Whether solver status was OPTIMAL_INACCURATE at each step.
    """

    step_times_min: list[int]
    step_times_hours: list[float]
    measured_feeder_kw: list[float]
    total_ev_power_kw: list[float]
    base_load_kw: list[float]
    transformer_limit_kw: float
    overload_kw: list[float]
    solver_statuses: list[str]
    sessions: list[EVSession]
    true_feeder_kw: list[float] = field(default_factory=list)
    alarms_raised: list[bool] = field(default_factory=list)
    safe_mode_active: list[bool] = field(default_factory=list)
    solvers_used: list[str | None] = field(default_factory=list)
    inaccurate_statuses: list[bool] = field(default_factory=list)
    is_protection_active: bool = False
    protection_active: list[bool] = field(default_factory=list)
    zero_power_unconfirmed: list[bool] = field(default_factory=list)
    unconfirmed_zero_sessions: list[list[str]] = field(default_factory=list)


def run_mpc_charging_loop(
    feeder: Feeder,
    sessions: list[EVSession],
    tariff: Tariff,
    meter: MeterInterface,
    evse: EVSEInterface,
    simulator: Any | None = None,
    time_step_min: int = 15,
    num_steps: int = 96,
    unmet_penalty_per_kwh: float = 100.0,
    horizon_steps: int | None = None,
    detector: Any | None = None,
    nominal_margin_kw: float = 1.0,
    conservative_margin_kw: float = 15.0,
    safe_mode_cooldown_steps: int = 4,
    use_telemetry_base_tracking: bool = True,
    max_aggregate_ev_kw: float | None = None,
    trusted_meter: MeterInterface | None = None,
) -> MPCSimulationResult:
    """Execute receding-horizon MPC EV charging simulation with optional cyber detector.

    At each discrete 15-minute time step:
    1. Reads the feeder meter through MeterInterface.
    2. Runs bad-data detector on energy balance: expected = base_forecast + sum(ev_cmd).
    3. If alarm raised, switches to safe mode:
       - Discards corrupted meter reading, restoring trustworthy base forecast.
       - Caps setpoints conservatively by reducing available transformer headroom.
    4. Re-solves the constrained CVXPY optimization over the remaining horizon.
    5. Dispatches only the first step's optimal power setpoints via EVSEInterface.
    6. Advances physical simulation state by one time step.

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
    simulator : Optional[Any], default None
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
    detector : Optional[Any], default None
        Cyber anomaly detector (e.g., CUSUMDetector or IsolationForestDetector).
    nominal_margin_kw : float, default 1.0
        Capacity derating margin applied to transformer limit in nominal mode to reject sensor noise [kW].
    conservative_margin_kw : float, default 15.0
        Capacity derating margin applied to transformer limit in safe mode [kW].
    safe_mode_cooldown_steps : int, default 4
        Number of steps controller remains in safe mode after alarm [15-min steps].
    use_telemetry_base_tracking : bool, default True
        Whether controller infers base load drift from meter telemetry when unalarmed.

    Returns
    -------
    MPCSimulationResult
        Trajectory history including true loads, overloads, and cyber telemetry.
    """
    step_times_min: list[int] = []
    step_times_hours: list[float] = []
    measured_feeder_kw: list[float] = []
    true_feeder_kw: list[float] = []
    total_ev_power_kw: list[float] = []
    base_load_kw: list[float] = []
    overload_kw: list[float] = []
    solver_statuses: list[str] = []
    solvers_used: list[str | None] = []
    inaccurate_statuses: list[bool] = []
    alarms_raised: list[bool] = []
    safe_mode_active: list[bool] = []

    # Ensure all sessions have resolved unique IDs and reset delivered energy
    resolve_session_ids(sessions)
    for s in sessions:
        s.delivered_energy_kwh = 0.0

    if simulator is not None and hasattr(simulator, "reset"):
        simulator.reset()

    base_profile = list(feeder.base_load_profile_kw)
    if len(base_profile) < num_steps:
        pad = base_profile[-1] if base_profile else 30.0
        base_profile.extend([pad] * (num_steps - len(base_profile)))

    price_profile = list(tariff.price_per_kwh_per_15min)
    if len(price_profile) < num_steps:
        pad_p = price_profile[-1] if price_profile else 0.15
        price_profile.extend([pad_p] * (num_steps - len(price_profile)))

    safe_mode_timer: int = 0
    safe_mode_latched: bool = False
    protection_active: list[bool] = []
    zero_power_unconfirmed: list[bool] = []
    unconfirmed_zero_sessions: list[list[str]] = []

    for step in range(num_steps):
        t_min = step * time_step_min
        step_times_min.append(t_min)
        step_times_hours.append(t_min / 60.0)

        # 1. Read and validate primary feeder meter via HAL
        primary_valid = True
        max_age_s = float(time_step_min * 60)
        primary_is_fresh = False

        if hasattr(meter, "is_reading_fresh"):
            try:
                primary_is_fresh = bool(meter.is_reading_fresh(t_min, max_age_s=max_age_s))
            except (RuntimeError, OSError, ValueError, TypeError):
                primary_is_fresh = False
        elif hasattr(meter, "get_reading_timestamp_s"):
            try:
                ts = meter.get_reading_timestamp_s(t_min)
                if ts is not None and np.isfinite(float(ts)):
                    age_s = float(t_min * 60) - float(ts)
                    primary_is_fresh = 0.0 <= age_s <= max_age_s
            except (RuntimeError, OSError, ValueError, TypeError):
                primary_is_fresh = False

        if getattr(meter, "is_stale", False):
            primary_is_fresh = False

        if not primary_is_fresh:
            logger.warning(
                "Primary meter freshness cannot be established or reading is stale at t=%d min; "
                "latching safe state and applying fallback.",
                t_min,
            )
            primary_valid = False

        meas_kw = 0.0
        try:
            raw_meas = meter.read_feeder_kw(t_min)
            meas_val = float(raw_meas)
            if not math.isfinite(meas_val) or meas_val < -0.05:
                logger.warning(
                    "Primary meter reading invalid (val=%s) at t=%d min; latching safe state.",
                    raw_meas,
                    t_min,
                )
                primary_valid = False
                meas_kw = float("nan")
            else:
                meas_kw = max(0.0, meas_val)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Primary meter read_feeder_kw query failed at t=%d min: %s; latching safe state.",
                t_min,
                e,
            )
            primary_valid = False
            meas_kw = float("nan")

        measured_feeder_kw.append(meas_kw)

        if not primary_valid:
            safe_mode_latched = True

        curr_base_kw = base_profile[step]
        base_load_kw.append(curr_base_kw)

        # 2. Inspect active EV charging power and evaluate bad-data detector
        # Strictly queries EVSEInterface, preserving Hardware Abstraction Layer
        current_ev_measured_kw = 0.0
        evse_pre_read_failed = False
        for s in sessions:
            try:
                raw_s_power = evse.read_session_kw(s.session_id)
                s_power = float(raw_s_power)
                if not math.isfinite(s_power) or s_power < -0.05:
                    logger.warning(
                        "EVSE readback invalid (val=%s) for session %s at t=%d min; rejecting reading and entering safe state.",
                        raw_s_power,
                        s.session_id,
                        t_min,
                    )
                    evse_pre_read_failed = True
                else:
                    current_ev_measured_kw += max(0.0, s_power)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "EVSE read_session_kw query failed for session %s at t=%d min: %s; entering safe state.",
                    s.session_id,
                    t_min,
                    e,
                )
                evse_pre_read_failed = True

        expected_feeder_kw = curr_base_kw + current_ev_measured_kw

        alarm_now = bool(not primary_valid or safe_mode_latched)
        if evse_pre_read_failed:
            alarm_now = True
            safe_mode_latched = True

        if detector is not None and primary_valid and math.isfinite(meas_kw):
            alert = detector.update(reported_feeder_kw=meas_kw, expected_feeder_kw=expected_feeder_kw)
            alarm_now = alarm_now or bool(alert.is_alarm)

        # Independent trusted meter protective check (defense-in-depth against adaptive attacks)
        trusted_kw: float | None = None
        trusted_base_kw: float | None = None
        trusted_valid_step = False
        if trusted_meter is not None:
            max_age_s = float(time_step_min * 60)
            is_fresh = False

            # Check explicit freshness contract via MeterInterface API
            if hasattr(trusted_meter, "is_reading_fresh"):
                try:
                    is_fresh = bool(trusted_meter.is_reading_fresh(t_min, max_age_s=max_age_s))
                except (RuntimeError, OSError, ValueError, TypeError):
                    is_fresh = False
            elif hasattr(trusted_meter, "get_reading_timestamp_s"):
                try:
                    ts = trusted_meter.get_reading_timestamp_s(t_min)
                    if ts is not None and np.isfinite(float(ts)):
                        age_s = float(t_min * 60) - float(ts)
                        is_fresh = 0.0 <= age_s <= max_age_s
                except (RuntimeError, OSError, ValueError, TypeError):
                    is_fresh = False

            # Also check legacy or explicit is_stale attribute if present
            if getattr(trusted_meter, "is_stale", False):
                is_fresh = False

            if not is_fresh:
                logger.warning(
                    "Trusted meter freshness cannot be established or reading is stale at t=%d min; "
                    "rejecting reading, entering defensive safe mode and zero-charging fallback.",
                    t_min,
                )
                alarm_now = True
                trusted_valid_step = False
            else:
                try:
                    raw_trusted = trusted_meter.read_feeder_kw(t_min)
                    if raw_trusted is None or not np.isfinite(raw_trusted) or raw_trusted < 0.0:
                        logger.warning(
                            "Trusted meter reading invalid (val=%s) at t=%d min; rejecting reading and entering safe mode.",
                            raw_trusted,
                            t_min,
                        )
                        alarm_now = True
                        trusted_valid_step = False
                    else:
                        trusted_kw = float(raw_trusted)
                        trusted_base_kw = max(0.0, trusted_kw - current_ev_measured_kw)
                        trusted_valid_step = True
                except (RuntimeError, OSError, ValueError, TypeError) as e:
                    logger.warning(
                        "Trusted meter query failed at t=%d min: %s; entering defensive safe mode.",
                        t_min,
                        e,
                    )
                    alarm_now = True
                    trusted_valid_step = False

            if trusted_kw is not None and trusted_base_kw is not None:
                # Detect unmodeled surge or telemetry spoofing:
                # 1. Total physical load exceeds transformer safety threshold:
                overload_threshold = feeder.transformer_limit_kw - conservative_margin_kw
                if trusted_kw > overload_threshold:
                    alarm_now = True
                    logger.warning(
                        "Trusted meter alert at t=%d min: physical load %.2f kW exceeds safety threshold %.2f kW",
                        t_min,
                        trusted_kw,
                        overload_threshold,
                    )
                # 2. Discrepancy between trusted physical meter and primary reported meter:
                if primary_valid and math.isfinite(meas_kw) and abs(trusted_kw - meas_kw) > 5.0:
                    alarm_now = True
                    logger.warning(
                        "Trusted meter discrepancy at t=%d min: trusted %.2f kW vs reported %.2f kW (Delta=%.2f kW)",
                        t_min,
                        trusted_kw,
                        meas_kw,
                        abs(trusted_kw - meas_kw),
                    )

        alarms_raised.append(alarm_now)

        if alarm_now:
            safe_mode_timer = safe_mode_cooldown_steps

        is_safe_mode = bool(safe_mode_latched or safe_mode_timer > 0)
        if safe_mode_timer > 0:
            safe_mode_timer -= 1
        safe_mode_active.append(is_safe_mode)

        # 3. Compute telemetry offset and effective transformer rating
        if is_safe_mode:
            # Safe mode: cap setpoints more conservatively by reserving safety margin
            effective_transformer_limit_kw = max(
                0.0, feeder.transformer_limit_kw - conservative_margin_kw
            )
            # Under safe mode: if trusted meter is available, use trusted base load;
            # otherwise discard corrupted primary meter and use forecast base load
            if trusted_meter is not None and trusted_base_kw is not None:
                telemetry_offset_kw = max(0.0, trusted_base_kw - curr_base_kw)
            else:
                telemetry_offset_kw = 0.0
        else:
            effective_transformer_limit_kw = max(
                0.0, feeder.transformer_limit_kw - nominal_margin_kw
            )
            if use_telemetry_base_tracking and primary_valid and math.isfinite(meas_kw):
                inferred_current_base = max(0.0, meas_kw - current_ev_measured_kw)
                telemetry_offset_kw = inferred_current_base - curr_base_kw
            else:
                telemetry_offset_kw = 0.0

        # Compute conservative available headroom and fallback caps
        active_max_aggregate_kw = max_aggregate_ev_kw
        step_protection_active = False

        if not primary_valid or safe_mode_latched:
            # Primary meter invalid or stale, or safe state latched:
            # apply documented zero-charge or independent-cap fallback
            if max_aggregate_ev_kw is not None:
                active_max_aggregate_kw = max_aggregate_ev_kw
                step_protection_active = True
            else:
                active_max_aggregate_kw = 0.0
                step_protection_active = False

        if trusted_meter is not None:
            if trusted_valid_step and trusted_base_kw is not None:
                trusted_headroom_kw = max(0.0, effective_transformer_limit_kw - trusted_base_kw)
                if active_max_aggregate_kw is None:
                    active_max_aggregate_kw = trusted_headroom_kw
                else:
                    active_max_aggregate_kw = min(active_max_aggregate_kw, trusted_headroom_kw)
                step_protection_active = True
            else:
                # Trusted meter was supplied but reading was invalid, stale, or unavailable.
                # Fall back to independent aggregate cap if provided, else zero-charging (0.0 kW)
                if max_aggregate_ev_kw is not None:
                    active_max_aggregate_kw = max_aggregate_ev_kw
                    step_protection_active = True
                else:
                    active_max_aggregate_kw = 0.0
                    step_protection_active = False
        elif max_aggregate_ev_kw is not None and primary_valid and not safe_mode_latched:
            step_protection_active = True

        # 4. Determine remaining horizon for re-solving
        rem_steps = num_steps - step
        plan_steps = rem_steps if horizon_steps is None else min(horizon_steps, rem_steps)

        # Prepare remaining profiles
        rem_base = [
            max(0.0, b + telemetry_offset_kw)
            for b in base_profile[step : step + plan_steps]
        ]
        if trusted_meter is not None and trusted_base_kw is not None and is_safe_mode:
            rem_base = [max(b, trusted_base_kw) for b in rem_base]

        rem_feeder = Feeder(
            transformer_limit_kw=effective_transformer_limit_kw,
            base_load_profile_kw=rem_base,
        )
        rem_prices = price_profile[step : step + plan_steps]
        rem_tariff = Tariff(price_per_kwh_per_15min=rem_prices)

        # Construct time-shifted sessions relative to current time t_min
        active_rem_sessions: list[EVSession] = []
        horizon_end_min = t_min + plan_steps * time_step_min
        for s in sessions:
            dwell_in_horizon = s.overlap_minutes(t_min, horizon_end_min)
            if dwell_in_horizon > 0.0 and s.remaining_energy_req_kwh > 1e-4:
                rem_energy = s.remaining_energy_req_kwh
                arr_rel = max(0, s.arrival_min - t_min)
                dep_rel = max(arr_rel + 1, s.departure_min - t_min)
            else:
                rem_energy = 0.0
                arr_rel = plan_steps * time_step_min + 10
                dep_rel = arr_rel + 10

            shifted_s = EVSession(
                arrival_min=arr_rel,
                departure_min=dep_rel,
                energy_req_kwh=rem_energy,
                max_rate_kw=s.max_rate_kw,
                session_id=s.session_id,
                delivered_energy_kwh=0.0,
            )
            active_rem_sessions.append(shifted_s)

        # 5. Solve CVXPY schedule over the remaining horizon
        opt_res = solve_charging_schedule(
            sessions=active_rem_sessions,
            feeder=rem_feeder,
            tariff=rem_tariff,
            time_step_min=time_step_min,
            num_steps=plan_steps,
            unmet_penalty_per_kwh=unmet_penalty_per_kwh,
            max_aggregate_ev_kw=active_max_aggregate_kw,
        )
        solver_statuses.append(opt_res.status)
        solvers_used.append(opt_res.solver_used)
        inaccurate_statuses.append(opt_res.is_inaccurate)

        if not opt_res.is_success and (active_max_aggregate_kw is None or active_max_aggregate_kw > 0.0):
            step_protection_active = False
        protection_active.append(step_protection_active)

        # 6. Extract and apply ONLY the first step's setpoints
        dt_hours = time_step_min / 60.0
        step_ev_power = 0.0
        prev_energies: list[float] = [s.delivered_energy_kwh for s in sessions]
        evse_actuation_failed = False
        step_zero_unconfirmed = False
        session_actual_powers: dict[str, float] = {}

        for i, s in enumerate(sessions):
            if opt_res.is_success and i < opt_res.power_schedule_kw.shape[0]:
                cmd_kw = float(opt_res.power_schedule_kw[i, 0])
            else:
                # Safe fallback if solver failed: zero or safe minimal rate
                cmd_kw = 0.0

            # Dispatch via EVSEInterface with exception handling
            setpoint_dispatched = False
            try:
                evse.set_setpoint_kw(s.session_id, cmd_kw)
                setpoint_dispatched = True
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "EVSE set_setpoint_kw failed for session %s at t=%d min: %s; entering safe state.",
                    s.session_id,
                    t_min,
                    e,
                )
                evse_actuation_failed = True

            # Read back delivered power via EVSEInterface with validation
            actual_kw = 0.0
            readback_ok = False
            try:
                raw_actual = evse.read_session_kw(s.session_id)
                actual_val = float(raw_actual)
                if not math.isfinite(actual_val) or actual_val < -0.05:
                    logger.warning(
                        "EVSE readback invalid (val=%s) for session %s at t=%d min; rejecting reading and entering safe state.",
                        raw_actual,
                        s.session_id,
                        t_min,
                    )
                    evse_actuation_failed = True
                else:
                    actual_kw = max(0.0, actual_val)
                    readback_ok = True
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "EVSE read_session_kw query failed for session %s at t=%d min: %s; entering safe state.",
                    s.session_id,
                    t_min,
                    e,
                )
                evse_actuation_failed = True

            # Treat significant command / readback mismatch as actuation failure
            if setpoint_dispatched and readback_ok and not np.isclose(cmd_kw, actual_kw, atol=0.1):
                logger.warning(
                    "EVSE actuation discrepancy for session %s at t=%d min: commanded %.2f kW != delivered %.2f kW; actuation failure.",
                    s.session_id,
                    t_min,
                    cmd_kw,
                    actual_kw,
                )
                evse_actuation_failed = True

            session_actual_powers[s.session_id] = actual_kw

        unconfirmed_sessions: list[str] = []
        if evse_actuation_failed:
            safe_mode_latched = True
            safe_mode_timer = safe_mode_cooldown_steps
            if alarms_raised:
                alarms_raised[-1] = True
            if safe_mode_active:
                safe_mode_active[-1] = True
            if protection_active:
                protection_active[-1] = False

            # Emergency cluster shutdown: attempt to stop all sessions in the same control step
            logger.error(
                "EVSE actuation failure detected at t=%d min; initiating emergency shutdown across all sessions.",
                t_min,
            )
            for s_stop in sessions:
                try:
                    evse.set_setpoint_kw(s_stop.session_id, 0.0)
                except Exception as stop_err:  # noqa: BLE001
                    logger.error(
                        "Emergency stop set_setpoint_kw(0.0) failed for session %s at t=%d min: %s",
                        s_stop.session_id,
                        t_min,
                        stop_err,
                    )

                confirmed_zero = False
                final_power_kw = 0.0
                try:
                    raw_stop = evse.read_session_kw(s_stop.session_id)
                    stop_val = float(raw_stop)
                    if math.isfinite(stop_val):
                        if abs(stop_val) <= 0.05:
                            confirmed_zero = True
                            final_power_kw = 0.0
                        else:
                            final_power_kw = max(0.0, stop_val)
                    else:
                        final_power_kw = 0.0
                except Exception as read_err:  # noqa: BLE001
                    logger.error(
                        "Emergency stop read_session_kw failed for session %s at t=%d min: %s",
                        s_stop.session_id,
                        t_min,
                        read_err,
                    )
                    final_power_kw = 0.0

                if not confirmed_zero:
                    unconfirmed_sessions.append(s_stop.session_id)
                    step_zero_unconfirmed = True
                    logger.error(
                        "Zero power cannot be confirmed for session %s at t=%d min (delivered power: %.2f kW)",
                        s_stop.session_id,
                        t_min,
                        final_power_kw,
                    )

                session_actual_powers[s_stop.session_id] = final_power_kw

        zero_power_unconfirmed.append(step_zero_unconfirmed)
        unconfirmed_zero_sessions.append(unconfirmed_sessions)

        for s in sessions:
            step_ev_power += session_actual_powers.get(s.session_id, 0.0)

        total_ev_power_kw.append(step_ev_power)

        # Ground truth power flow: read from physical simulator when present
        if simulator is not None and hasattr(simulator, "get_feeder_power_kw"):
            actual_total_feeder = float(simulator.get_feeder_power_kw(t_min))
        else:
            actual_total_feeder = curr_base_kw + step_ev_power

        true_feeder_kw.append(actual_total_feeder)
        overload = max(0.0, actual_total_feeder - feeder.transformer_limit_kw)
        overload_kw.append(overload)

        # 7. Advance simulator by one time step if present
        if simulator is not None and hasattr(simulator, "step"):
            simulator.step()

        # Feedback state estimation: update delivered energy from measured EVSE submetering
        # if not already mutated by physical simulator
        for i, s in enumerate(sessions):
            if np.isclose(s.delivered_energy_kwh, prev_energies[i]):
                actual_s_kw = session_actual_powers.get(s.session_id, 0.0)
                if math.isfinite(actual_s_kw) and actual_s_kw >= 0.0:
                    s.delivered_energy_kwh += actual_s_kw * dt_hours

    overall_protection_active = bool(all(protection_active)) if protection_active else False
    if trusted_meter is None and max_aggregate_ev_kw is None:
        overall_protection_active = False

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
        true_feeder_kw=true_feeder_kw,
        alarms_raised=alarms_raised,
        safe_mode_active=safe_mode_active,
        solvers_used=solvers_used,
        inaccurate_statuses=inaccurate_statuses,
        is_protection_active=overall_protection_active,
        protection_active=protection_active,
        zero_power_unconfirmed=zero_power_unconfirmed,
        unconfirmed_zero_sessions=unconfirmed_zero_sessions,
    )
