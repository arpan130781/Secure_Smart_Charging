# Secure Smart EV Charging: Mitigating Feeder Meter FDI Attacks

A modular, reproducible cyber-physical research framework studying how **False Data Injection (FDI)** attacks on substation feeder meter readings degrade **Model Predictive Control (MPC)** charging schedulers, and evaluating **lightweight physics-informed and machine learning detectors** that restore grid security and charging performance.

Designed for an academic interview with a university professor working on EV charging security and power grid analytics. Every variable, metric, and equation is grounded in physical principles with explicit units ($kW$, $kWh$, $minutes$).

---

## 1. Problem Formulation & Cyber Vulnerability

In modern distribution networks, clusters of Electric Vehicle Supply Equipment (EVSE) are coordinated by a central Energy Management System (EMS). To avoid overloading distribution transformers and thermal cable ratings, the EMS uses **Model Predictive Control (MPC)** to schedule charging within available capacity:

$$\sum_{i \in \text{EVs}} P_i(t) \le P_{\text{feeder}}^{\max} - \hat{P}_{\text{base}}(t) - P_{\text{margin}}$$

where:
- $P_{\text{feeder}}^{\max}$: Transformer continuous thermal capacity rating [kW] (e.g. 120.0 kW).
- $\hat{P}_{\text{base}}(t)$: Unmanaged background grid load (residential/commercial demand) [kW].
- $P_i(t)$: Commanded active charging power dispatched to EV $i$ [kW].
- $P_{\text{margin}}$: Nominal headroom buffer reserved against sensor noise [kW] (e.g. 1.0 kW $\approx 3.3\sigma$ for $\sigma = 0.3\text{ kW}$).

### Synthetic Session Generation (Inhomogeneous Poisson Process)
Fleet arrival times are modeled via a Non-Homogeneous Poisson Process (NHPP) conditioned on total fleet size $N$, drawing arrival slots via multinomial sampling from a normalized diurnal intensity profile $f(t) = \lambda(t) / \int \lambda(u)\,du$ with a morning commuting peak centered at 08:30 ($\mu = 510\text{ min}$, $\sigma = 75\text{ min}$). Dwell durations follow a lognormal distribution, and energy requests follow a truncated normal distribution bounded by battery capacity.

### The Cyber Vulnerability (FDI on Feeder Telemetry)
Substation feeder meters communicate with the EMS over SCADA or AMI telemetry links (e.g. Modbus/TCP, DNP3, or cellular backhaul). Because these legacy links often lack end-to-end cryptographic integrity or authentication, an adversary can intercept and manipulate the telemetry:
1. **Under-Reporting Attack (Feeder Overload):** Falsely reporting lower feeder load ($y_{\text{reported}}(t) = P_{\text{feeder}}(t) - \Delta$) misleads the MPC scheduler into perceiving phantom headroom. The scheduler commands peak EV charging, pushing physical power flow well beyond transformer thermal ratings ($P_{\text{feeder}}(t) \gg P_{\text{feeder}}^{\max}$), causing accelerated thermal aging, insulation breakdown, or protection trips.
2. **Stealthy Ramp Attack:** The adversary slowly increases $\Delta(t) = \alpha \cdot t$ to bypass naive rate-of-change thresholds.
3. **Replay Attack:** Repeating low-load nocturnal telemetry during peak daytime arrival hours creates massive capacity miscalculations.

---

## 2. System Block Diagram (Plain ASCII)

The system maintains strict architectural separation between physical ground-truth physics and cyber control logic via a **Hardware Abstraction Layer (HAL)**:

```
+-----------------------------------------------------------------------------------------+
|                                  PHYSICAL ENVIRONMENT                                   |
|                                                                                         |
|  - Substation Transformer Rating: P_feeder_max = 120.0 kW                               |
|  - Diurnal Non-EV Base Load: P_base(t) [SIMULATED GROUND TRUTH]                         |
|  - EV Fleet Battery States: E_i(t+1) = E_i(t) + eta * P_i(t) * dt [SIMULATED]           |
|  - True Feeder Active Power: P_feeder(t) = P_base(t) + sum_i(P_i(t)) [SIMULATED]        |
+--------------------------------------------+--------------------------------------------+
                                             |
                          +------------------+------------------+
                          |                                     |
                          v                                     v
           +-----------------------------+       +-----------------------------+
           |     hal.MeterInterface      |       |      hal.EVSEInterface      |
           |   - read_feeder_kw(t_min)   |       |   - set_setpoint_kw(ev, P)  |
           |   [MEASURED Feeder Power]   |       |   - read_session_kw(ev)     |
           |   (Subject to Noise & FDI)  |       |   [MEASURED Submetering]    |
           +--------------+--------------+       +--------------+--------------+
                          |                                     |
                          | (Attacked Telemetry)                | (Trusted Submetering)
                          v                                     |
           +---------------------------------------------+      |
           |            CYBER FAULT DETECTOR             |      |
           |  - Innovation: r(t) = y_meter - (P_base + P) |      |
           |  - Detector 1: Physics CUSUM (S^-)          |<-----+
           |  - Detector 2: Isolation Forest [r, dr, r/e]|      |
           +----------------------+----------------------+      |
                                  |                             |
                                  | Alarm Trigger               |
                                  v                             |
           +---------------------------------------------+      |
           |              SAFE MODE SWITCH               |      |
           |  - If Alarm: Discard meter reading;         |      |
           |    revert to base load expectation P_base.  |      |
           |  - Derate capacity: P_limit - P_margin      |      |
           +----------------------+----------------------+      |
                                  |                             |
                                  | Clean / Restored Headroom   |
                                  v                             |
           +---------------------------------------------+      |
           |           RECEDING-HORIZON MPC              |      |
           |  - CVXPY Convex Program (Clarabel / OSQP)   |      |
           |  - Optimizes costs subject to feeder limit  |      |
           |  - Dispatches first-step setpoints P_i^*(t) |------+
           +---------------------------------------------+
```

---

## 3. Simulated vs. Real (Ground Truth vs. Telemetry Disambiguation)

To preserve scientific validity, all variables in code, logs, and interfaces are strictly classified into three categories:

| Variable | Implementation Location | Category | Physical Meaning & Engineering Units |
|---|---|---|---|
| `sim_base_load_kw` | `Feeder.base_load_profile_kw` | **[SIMULATED]** | Ground-truth unmanaged residential/commercial building active power [kW]. |
| `sim_feeder_total_power_kw`| `ChargingStationSimulator` | **[SIMULATED]** | Ground-truth physical active power entering the distribution substation [kW]. |
| `sim_overload_kw` | `ChargingStationSimulator` | **[SIMULATED]** | Ground-truth thermal violation $\max(0, P_{\text{feeder}} - P_{\text{feeder}}^{\max})$ [kW]. |
| `delivered_energy_kwh` | `EVSession.delivered_energy_kwh`| **[SIMULATED]** | Ground-truth electrochemical energy stored in battery pack [kWh]. |
| `meas_feeder_kw` | `MeterInterface.read_feeder_kw` | **[MEASURED]** | Telemetry reported by substation meter; contains noise $\mathcal{N}(0, \sigma^2)$ and is subject to FDI [kW]. |
| `meas_session_kw` | `EVSEInterface.read_session_kw` | **[MEASURED]** | Submetered active charging power at individual on-premises charging stalls [kW]. |
| `inferred_base_kw` | `run_mpc_charging_loop` | **[DERIVED]** | Telemetry-derived unmanaged base load: $\max(0, y_{\text{meter}} - \sum P_{\text{ev}}^{\text{meas}})$ [kW]. |
| `residual_kw` | `DetectorAlert.residual_kw` | **[DERIVED]** | Innovation residual $r(t) = y_{\text{meter}}(t) - (\hat{P}_{\text{base}}(t) + \sum P_i(t))$ [kW]. |
| `restored_feeder_kw` | `run_mpc_charging_loop` | **[DERIVED]** | Safe mode synthesized feeder load estimate used when meter is distrusted [kW]. |

### Invariant Enforcement
Controller and detector logic **never read the simulator directly**. All inputs are acquired strictly via `MeterInterface` and `EVSEInterface`, guaranteeing that the algorithmic layer operates purely on physical measurements and model forecasts.

---

## 4. How to Reproduce Every Figure with One Command Each

All experiments are deterministic and fully reproducible from seeded YAML configs using the provided `Makefile`:

| Figure / Artifact | Command | Description |
|---|---|---|
| **Setup Environment** | `make setup` | Creates virtualenv and installs dependencies (`pip install -e ".[dev]"`). |
| **Run Test Suite** | `make test` | Executes all 39 pytest unit and integration tests. |
| **Baseline Figures** | `make baseline` | Generates `results/baseline_load.png` and `results/baseline_vs_mpc.png` (Task 5 & 8). |
| **Attack Impact Figure** | `make attack` | Simulates 80 MPC runs across 20 seeds; generates `results/attack_impact.png` and `results/attack_impact.csv` (Task 11). |
| **Detector Evaluation** | `make detect` | Evaluates CUSUM and Isolation Forest detectors; outputs `results/detection_metrics.csv` (Task 14). |
| **Headline Figure** | `make headline` | Simulates 24-hour comparative run; generates `results/headline.png` and `results/headline.csv` (Task 16). |

### Direct CLI Commands (Without Make)
```bash
# 1. Baseline vs MPC comparison (20 seeds)
PYTHONPATH=src:. python experiments/compare_baseline_vs_mpc.py

# 2. Attack impact & detector evaluation (20 seeds, 4 conditions)
PYTHONPATH=src:. python experiments/run_attack_and_detection_experiments.py

# 3. Headline 24-hour comparative simulation
PYTHONPATH=src:. python experiments/run_headline_experiment.py

# 4. Adaptive attacker limitation experiment
PYTHONPATH=src:. python experiments/run_adaptive_attacker_limitation.py
```

---

## 5. Summary of Empirical Results

All results are reported with statistical spreads ($\text{mean} \pm \text{std}$) and 95% confidence intervals across independent seeded simulation runs.

### 5.1 Baseline (Uncontrolled) vs. Optimal MPC (20 Seeds)
Evaluated over 20 random seeds (Seeds 100 to 119; 20–50 EVs per day, NHPP arrival with morning peak, lognormal dwell times, truncated normal energy requests; generated via `experiments/compare_baseline_vs_mpc.py`):

| Metric | Uncontrolled Baseline | MPC Scheduler (Clean) | Improvement |
|---|:---:|:---:|:---:|
| **Peak Feeder Load [kW]** | **$217.09 \pm 24.65\text{ kW}$** | **$119.00 \pm 0.00\text{ kW}$** | **-45.2%** (strictly below 120.0 kW limit) |
| **Overload Duration [min]** | **$252.00 \pm 26.03\text{ min}$** | **$0.00 \pm 0.00\text{ min}$** | **-100.0%** (complete overload elimination across all seeds) |
| **Total Energy Cost [\$]** | **$\$160.72 \pm \$6.89$** | **$\$157.42 \pm \$7.12$** | **-2.1%** (\$3.30/day cost savings) |
| **Unmet Energy [kWh]** | **$36.28 \pm 15.94\text{ kWh}$** | **$68.64 \pm 43.62\text{ kWh}$** | Controlled curtailment to protect transformer |

*95% Confidence Intervals (20 seeds):*
- Uncontrolled Peak: $[206.29, 227.89]\text{ kW}$ vs. MPC Peak: $[119.00, 119.00]\text{ kW}$
- Uncontrolled Overload: $[240.59, 263.41]\text{ min}$ vs. MPC Overload: $[0.00, 0.00]\text{ min}$
- Uncontrolled Cost: $[\$157.70, \$163.74]$ vs. MPC Cost: $[\$154.30, \$160.54]$

### 5.2 Cyber Attack Impact on Grid Safety (50 Seeds, 200 Simulations)
Evaluated across 50 independent seeds (Seeds 200 to 249; 200 total MPC simulations; FDI telemetry attack injected between 08:00 and 14:00; generated via `experiments/run_attack_and_detection_experiments.py`):

| Cyber Condition | Peak True Load [kW] | Transformer Overload [min] | Electricity Cost [\$] | Unmet Energy [kWh] |
|---|:---:|:---:|:---:|:---:|
| **No Attack (Clean MPC)** | $120.38 \pm 0.10\text{ kW}$ | $2.70 \pm 5.82\text{ min}$ | $\$156.87 \pm \$10.03$ | $70.56 \pm 36.44\text{ kWh}$ |
| **Constant Bias ($-35\text{ kW}$)** | **$144.55 \pm 3.36\text{ kW}$** (+24.2 kW) | **$297.00 \pm 36.62\text{ min}$** (+294.3 min) | $\$160.91 \pm \$8.23$ | $37.28 \pm 21.27\text{ kWh}$ |
| **Ramp Attack ($0 \to -35\text{ kW}$)** | **$140.15 \pm 4.76\text{ kW}$** (+19.8 kW) | **$319.50 \pm 25.58\text{ min}$** (+316.8 min) | $\$159.60 \pm \$8.17$ | $45.18 \pm 24.75\text{ kWh}$ |
| **Replay Attack (Valley Replay)** | **$144.49 \pm 3.40\text{ kW}$** (+24.1 kW) | **$297.00 \pm 35.47\text{ min}$** (+294.3 min) | $\$160.90 \pm \$8.23$ | $37.28 \pm 21.27\text{ kWh}$ |

*95% Confidence Intervals (50 seeds):*
- Constant Bias Overload: $[286.85, 307.15]\text{ min}$
- Ramp Attack Overload: $[312.41, 326.59]\text{ min}$
- Replay Attack Overload: $[287.17, 306.83]\text{ min}$

### 5.3 Detector Performance (Precision, Recall, FAR, Delay across 50 Seeds)
Evaluated across 162.5 clean operational day equivalents:

| Detector | Attack Type | Precision | Recall | False Alarms / Day | Detection Delay [min] |
|---|---|:---:|:---:|:---:|:---:|
| **CUSUM (Physics Residual)** | Constant Bias | **1.000** | **1.000** | **0.00** | **0.0 min** |
| **CUSUM (Physics Residual)** | Ramp Attack | **1.000** | **0.583\*** | **0.00** | **90.0 min** |
| **CUSUM (Physics Residual)** | Replay Attack | **1.000** | **0.992** | **0.00** | **1.2 min** |
| **Isolation Forest (ML)** | Constant Bias | **0.993** | **0.125** | 0.03 | **0.0 min** |
| **Isolation Forest (ML)** | Ramp Attack | **0.999** | **0.958** | 0.03 | **15.0 min** (Fast $\Delta r$ detection) |
| **Isolation Forest (ML)** | Replay Attack | **0.999** | **0.947** | 0.03 | **0.0 min** |

*\*Note on Ramp Recall: Step-level classification recall is 58.3% because early ramp steps ($\Delta < 4.3\text{ kW}$) lie within normal sensor noise; however, episode-level detection is 100% across all 50 seeds well before thermal limits are compromised.*

### 5.4 Headline Mitigation Comparison (`results/headline.csv`)
*Single representative 24-hour trace (`seed=42`) used for publication time-series figure `results/headline.png`:*

| Operational Condition | Peak True Feeder Load [kW] | Transformer Overload [min] | Total Cost [\$] | Unmet Energy [kWh] |
|---|:---:|:---:|:---:|:---:|
| **Normal Operation (Clean MPC)** | 119.50 kW | 0.0 min | \$156.15 | 78.10 kWh |
| **Under Attack (Unmitigated)** | **140.82 kW** (+21.3 kW) | **300.0 min** (+300.0 min) | \$159.38 | 43.52 kWh |
| **Under Attack with Detection (Mitigated)** | **118.66 kW** (-22.2 kW) | **0.0 min** (-300.0 min) | \$144.71 | 135.96 kWh |

---

## 6. Limitations & Academic Discussion

### 6.1 Vulnerability to Model-Cognizant Adaptive Attackers (Task 17)
The energy-balance residual check relies on:
$$r(t) = y_{\text{reported}}(t) - \left(P_{\text{base}}^{\text{forecast}}(t) + \sum_{i} P_i^{\text{cmd}}(t)\right)$$
If an adversary acquires knowledge of the EMS base load forecast $P_{\text{base}}^{\text{forecast}}(t)$ and observes dispatched setpoints $\sum P_i(t)$, they can synthesize telemetry that dynamically satisfies the expected balance:
$$y_{\text{reported}}(t) = P_{\text{base}}^{\text{forecast}}(t) + \sum_{i} P_i^{\text{cmd}}(t) + \mathcal{N}(0, \sigma^2)$$

**Empirical Result (`experiments/run_adaptive_attacker_limitation.py`):**
When an unmodeled physical load surge increases actual base demand from $45\text{ kW}$ to $75\text{ kW}$ under an $80\text{ kW}$ transformer limit:
- True feeder load reaches **$90.00\text{ kW}$** (**$+10.0\text{ kW}$ overload**) for **120.0 minutes**.
- Innovation residual remains zero-mean noise ($|r(t)| \le 0.5\text{ kW}$).
- Detector 1 CUSUM raises **0 alarms (100% missed detection rate)**.

### 6.2 Active Power vs. Distribution Grid Physics
- **Active Power Simplification:** Feeder limits are modeled strictly via active power ($P$ in kW). In actual distribution feeders with high resistance-to-reactance ratios ($R/X \ge 2$), active power injections create localized voltage drops ($V < 0.95\text{ p.u.}$) and phase unbalance before transformer thermal ampacity is reached. Future extensions require Second-Order Cone Programming (SOCP) for branch power flow.
- **Submeter Trust Asymmetry:** The framework assumes EVSE internal submeters communicate over authenticated TLS/OCPP links that remain uncompromised. If an adversary compromises the EVSE gateway, the energy balance invariant is defeated.

### 6.3 Cyber-Physical Assumptions: Simulated vs. Real Hardware Realities
When transitioning from the software simulator to physical hardware deployment (e.g. Raspberry Pi + ATM90E26 + OCPP 1.6J), several foundational simulation assumptions must be accounted for:
1. **Instantaneous Setpoint Actuation:** The simulation models step setpoint tracking as instantaneous. In reality, on-board chargers (OBC) and battery contactors require 5 to 30 seconds to ramp active power due to PWM pilot negotiation (IEC 61851), DC precharge, and internal current control loop settling.
2. **Synchronous 15-Minute Periodic Clock:** The simulation operates on an idealized discrete clock ($t = 0, 15, 30, \dots$). In physical deployments, microcontrollers and smart meters experience oscillator clock drift and network jitter ($\pm 1\text{--}10\text{ s}$), requiring asynchronous sample timestamping.
3. **Lossless Telemetry Links:** Modbus/TCP, DNP3, or cellular backhaul links suffer occasional packet drops ($1\text{--}5\%$) and transmission latencies ($50\text{--}2000\text{ ms}$). The HAL must implement sample-and-hold buffers or NaN-tolerant estimation.
4. **Charger Efficiency & Power Factor:** The simulator assumes 100% conversion efficiency ($\eta = 1.0$) and unity power factor ($\text{PF} = 1.0$). Physical Level-2 chargers exhibit efficiencies of $88\%\text{--}94\%$ and draw reactive power ($Q$ in kVAR).
5. **Intra-Step Base Load Invariance:** Non-EV base demand is modeled as piecewise constant over 15-minute intervals. In reality, customer appliances fluctuate continuously within each interval, increasing the variance of the innovation residual $r(t)$.

---

## 7. Hardware Deployment Roadmap & CAN Bus Security

- **Physical Metering & EVSE Interfaces:** Detailed hardware transition guide in [`docs/hardware_path.md`](docs/hardware_path.md) (Microchip ATM90E26 SPI / TI INA219 I2C drivers and OCPP 1.6-J JSON over WebSockets).
- **Vehicle-to-Charger CAN Security:** Independent CAN bus intrusion detection module in [`can_security/`](can_security/) evaluating timing anomalies ($\Delta t \ne 100\text{ ms}$) and physical payload range violations.
