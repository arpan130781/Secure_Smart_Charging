# Secure Smart EV Charging: Feeder Meter FDI Degradation & Lightweight Detection

A modular, reproducible cyber-physical research framework studying how **False Data Injection (FDI)** attacks on substation feeder meter readings degrade a **Model Predictive Control (MPC)** EV charging scheduler, and evaluating **lightweight, physics-informed and machine learning detectors** that restore grid security.

Designed for an academic interview with a university professor working on EV charging security and power grid analytics. Every variable, metric, and equation is grounded in physical principles with explicit units ($kW$, $kWh$, $minutes$).

---

## 1. Problem Motivation & Cyber-Physical Threat Model

In modern distribution networks, clusters of Electric Vehicle Supply Equipment (EVSE) are coordinated by a central Energy Management System (EMS). To avoid overloading distribution transformers and thermal cable ratings, the EMS uses **Model Predictive Control (MPC)** to schedule charging within available capacity:

$$\sum_{i \in \text{EVs}} P_i(t) \le P_{\text{feeder}}^{\max} - \hat{P}_{\text{base}}(t)$$

where:
- $P_{\text{feeder}}^{\max}$: Transformer thermal capacity limit [kW].
- $\hat{P}_{\text{base}}(t)$: Unmanaged background grid load (residential/commercial) [kW].
- $P_i(t)$: Active charging power dispatched to EV $i$ [kW].

### The Cyber Vulnerability (FDI on Feeder Telemetry)
Substation feeder meters communicate with the EMS over SCADA or AMI telemetry links (e.g., Modbus/TCP, DNP3, or wireless backhaul). If an adversary tampers with this telemetry:
1. **Under-Reporting Attack (Feeder Overload):** Falsely reporting lower feeder load ($y_{\text{feeder}}(t) = P_{\text{feeder}}(t) - \Delta$) misleads the MPC scheduler into perceiving phantom headroom. The scheduler commands full-rate EV charging, forcing the true physical transformer load into severe overload ($P_{\text{feeder}}(t) \gg P_{\text{feeder}}^{\max}$), causing accelerated transformer loss-of-life and thermal degradation.
2. **Gradual Ramp Attack:** The adversary slowly drifts $\Delta(t) = \alpha \cdot t$ to bypass naive rate-of-change thresholds.
3. **Replay Attack:** Repeating low-load nocturnal telemetry during peak evening arrival hours creates massive capacity miscalculations.

---

## 2. Cyber-Physical Architecture & Hardware Abstraction Layer (HAL)

To ensure modularity and realism, controller and detector logic **never inspect the simulator directly**. All interactions pass through abstract interfaces in `src/ev_sched/hal.py`, allowing real OCPP gateways or physical smart meters to replace the simulator without altering algorithmic code.

```
+--------------------------------------------------------------------------------+
|                             PHYSICAL ENVIRONMENT                               |
|  - Distribution Transformer (P_feeder_max = 120 kW)                            |
|  - Diurnal Non-EV Base Load P_base(t) [SIMULATED]                              |
|  - EV Battery Electrochemical State E_i(t+1) = E_i(t) + P_i(t)*dt [SIMULATED]   |
|  - Feeder Physical Power Flow P_feeder(t) = P_base(t) + sum(P_i(t)) [SIMULATED]|
+---------------------------------------+----------------------------------------+
                                        |
                   +--------------------+--------------------+
                   |                                         |
                   v                                         v
    +------------------------------+          +------------------------------+
    |      hal.MeterInterface      |          |      hal.EVSEInterface       |
    |  - read_feeder_kw(t_min)     |          |  - set_setpoint_kw(ev_id, P) |
    |  [MEASURED Telemetry]        |          |  - read_session_kw(ev_id)    |
    |  (Subject to noise & FDI)    |          |  [MEASURED Submetering]      |
    +--------------+---------------+          +--------------+---------------+
                   |                                         |
                   |       +-------------------------+       |
                   +------>|   CYBER FAULT DETECTOR  |<------+
                   |       | - Detector 1: CUSUM     |
                   |       | - Detector 2: IsoForest |
                   |       +-------------------------+
                   v
    +------------------------------------------------------------------------+
    |                         MPC CHARGING SCHEDULER                         |
    |  - CVXPY convex quadratic/linear program (Clarabel / OSQP)             |
    |  - Incurs no knowledge of simulated ground truth                       |
    |  - Telemetry-derived base load estimate + Time-of-Use tariff           |
    +------------------------------------------------------------------------+
```

### Simulated vs Measured Variable Disambiguation

| Variable | Scope | Status | Physical Meaning & Units |
|---|---|---|---|
| `sim_feeder_power_kw` | `ChargingStationSimulator` | **[SIMULATED]** | True physical aggregate power flowing through transformer [kW] |
| `sim_ev_power_kw` | `ChargingStationSimulator` | **[SIMULATED]** | True electrochemical power entering vehicle batteries [kW] |
| `sim_base_load_kw` | `Feeder.base_load_profile_kw` | **[SIMULATED]** | True non-EV background demand [kW] |
| `sim_remaining_req_kwh` | `EVSession` | **[SIMULATED]** | True remaining energy needed by vehicle [kWh] |
| `meas_feeder_kw` | `MeterInterface` | **[MEASURED]** | Telemetry reported by substation meter (prone to noise & FDI) [kW] |
| `meas_session_kw` | `EVSEInterface` | **[MEASURED]** | Submetered active charging power at individual charging stall [kW] |
| `inferred_base_kw` | `Scheduler` / `Detector` | **[DERIVED]** | Telemetry-derived base load $\max(0, y_{\text{feeder}} - \sum P_{\text{ev}}^{\text{meas}})$ [kW] |
| `residual_kw` | `DetectorAlert` | **[DERIVED]** | Innovation residual $r(t) = y_{\text{feeder}}(t) - (P_{\text{base}}^{\text{forecast}} + \sum P_i)$ [kW] |

---

## 3. Mathematical Formulations

### 3.1 CVXPY MPC Scheduling Formulation
At each 15-minute decision step $k$, with lookahead horizon $H$, the scheduler solves:

$$\min_{\mathbf{P}, \mathbf{s}} \sum_{t=0}^{H-1} \sum_{i \in \mathcal{E}} c(t) \cdot P_{i,t} \cdot \Delta t + \lambda_{\text{slack}} \sum_{i \in \mathcal{E}} s_i + \lambda_{\text{smooth}} \sum_{t=0}^{H-2} \sum_{i \in \mathcal{E}} (P_{i,t+1} - P_{i,t})^2$$

**Subject to:**
1. **Transformer Feeder Capacity Limit:**
   $$\sum_{i \in \mathcal{E}} P_{i,t} + \hat{P}_{\text{base}}(t) \le P_{\text{transformer}}^{\max}, \quad \forall t \in [0, H-1]$$
2. **Per-EV Power Saturation:**
   $$0 \le P_{i,t} \le P_i^{\max} \cdot \mathbf{1}_{\{t_{\text{arr}, i} \le t < t_{\text{dep}, i}\}}, \quad \forall i, t$$
3. **Soft Energy Fulfillment with Slack ($s_i \ge 0$):**
   $$\sum_{t=0}^{H-1} P_{i,t} \cdot \Delta t + s_i \ge E_i^{\text{req}}, \quad \forall i \in \mathcal{E}$$

Here $\Delta t = 0.25\text{ h}$ (15 min), $c(t)$ is the Time-of-Use tariff [\$/kWh], and $\lambda_{\text{slack}} = 1000.0$ penalizes unmet charging demands.

### 3.2 False Data Injection Attack Models
All attacks wrap a `MeterInterface`:
- **(a) Constant Bias:** $y(t) = \max(0, P_{\text{true}}(t) - \Delta)$ during $t \in [t_{\text{start}}, t_{\text{start}} + T]$.
- **(b) Ramp Attack:** $y(t) = \max\left(0, P_{\text{true}}(t) - \Delta_{\max} \cdot \frac{t - t_{\text{start}}}{T}\right)$ stealthily ramping from 0 to $\Delta_{\max}$.
- **(c) Replay Attack:** $y(t) = P_{\text{buffer}}(t \pmod W)$ replaying a nocturnal valley window.

### 3.3 Cyber Detectors
#### Detector 1: Physics-Informed Residual CUSUM
Exploits conservation of power:
$$r(t) = y_{\text{feeder}}(t) - \left(\hat{P}_{\text{base}}(t) + \sum_{i} P_i(t)\right)$$
Under nominal operation, $r(t) \sim \mathcal{N}(0, \sigma^2)$ ($\sigma \approx 0.3\text{ kW}$).
A one-sided downward CUSUM accumulator tracks under-reporting attacks:
$$S^-(t) = \max(0, S^-(t-1) - r(t) - \nu)$$
- **Slack parameter $\nu = 1.5\text{ kW}$** ($\approx 5\sigma$): Guarantees $\mathbb{E}[-r(t) - \nu] < 0$ under $H_0$, preventing false accumulation.
- **Threshold $h = 18.0\text{ kW}$**: Chosen via Page's Average Run Length (ARL) theory to yield zero false alarms per day ($\text{ARL}_0 > 10^5$ steps) while flagging an injection within 1-2 steps.

#### Detector 2: Isolation Forest Anomaly Detector
An unsupervised ensemble of 100 isolation trees trained exclusively on attack-free operational runs ($N=480$ steps). Evaluates the 3-dimensional feature vector at each step:
$$\mathbf{x}(t) = \left[ r(t), \; \Delta r(t), \; \frac{y_{\text{feeder}}(t)}{\max(1, P_{\text{expected}}(t))} \right]$$

---

## 4. Empirical Evaluation & Results

### 4.1 Baseline vs. MPC Charging (20 Seeds)
Simulated across 20 distinct random seeds (20-50 EVs, Poisson arrivals, lognormal dwell times, truncated normal energy requests):

| Metric | Uncontrolled Baseline | MPC Scheduler (Clean) | Improvement |
|---|:---:|:---:|:---:|
| **Peak Feeder Load [kW]** | **172.93 kW** | **120.62 kW** | **-30.2%** (safely constrained to 120 kW limit) |
| **Overload Duration [min]** | **233.2 min** | **140.2 min\*** | **-39.9%** |
| **Overload Energy [kWh]** | **78.41 kWh** | **0.86 kWh** | **-98.9%** (eliminated transformer stress) |
| **Total Energy Cost [\$]** | **\$179.31** | **\$149.47** | **-16.6%** (\$29.84/day cost savings) |
| **Unmet Energy [kWh]** | **0.00 kWh** | **100.42 kWh** | Managed tradeoff during high-load peaks |

*\*Note: Clean MPC peak load of $120.62\text{ kW}$ is within the $\pm 0.9\text{ kW}$ ($3\sigma$) sensor measurement noise band around the $120.0\text{ kW}$ limit.*

### 4.2 Attack Impact on Grid Safety (20 Seeds, 80 Simulations)
Telemetry under-reporting between 09:00 and 16:00 (steps 36 to 64):

| Cyber Condition | Peak True Load [kW] | Transformer Overload [min] | Electricity Cost [\$] | Unmet Energy [kWh] |
|---|:---:|:---:|:---:|:---:|
| **No Attack (Clean MPC)** | 120.62 kW | 140.2 min | \$149.47 | 100.42 kWh |
| **Constant Bias ($-35\text{ kW}$)** | **147.70 kW** (+27.7 kW overload!) | **291.8 min** (+151.6 min overload) | \$157.95 | 55.45 kWh |
| **Ramp Attack ($0 \to -35\text{ kW}$)** | **139.08 kW** (+19.1 kW overload!) | **312.0 min** (+171.8 min overload) | \$154.44 | 73.15 kWh |
| **Replay Attack (Valley Replay)** | **147.70 kW** (+27.7 kW overload!) | **291.8 min** (+151.6 min overload) | \$157.95 | 55.45 kWh |

**Key Academic Finding:** Under all three FDI attacks, the scheduler acts on false headroom and commands excessive charging, driving true transformer load up to **$147.70\text{ kW}$** (**23.1% above thermal capacity**) and doubling transformer overload duration.

### 4.3 Detector Performance (Precision, Recall, FAR, Delay)

| Detector | Attack Type | Precision | Recall | False Alarms / Day | Detection Delay [min] |
|---|---|:---:|:---:|:---:|:---:|
| **CUSUM (Physics Residual)** | Constant Bias | **1.000** | **1.000** | **0.00** | **0.0 min** (Immediate) |
| **CUSUM (Physics Residual)** | Ramp Attack | **1.000** | **0.590** | **0.00** | **90.0 min** (Waits for drift to exceed $h$) |
| **CUSUM (Physics Residual)** | Replay Attack | **1.000** | **0.973** | **0.00** | **0.8 min** (Near-immediate) |
| **Isolation Forest (ML)** | Constant Bias | **0.990** | **1.000** | 0.33 | **0.0 min** (Immediate) |
| **Isolation Forest (ML)** | Ramp Attack | **0.989** | **0.950** | 0.33 | **18.0 min** (Fast $\Delta r$ detection) |
| **Isolation Forest (ML)** | Replay Attack | **0.990** | **1.000** | 0.33 | **0.0 min** (Immediate) |

**Key Comparison:**
- **CUSUM:** Zero false alarms ($\text{FAR} = 0.0/\text{day}$), perfect precision (1.0), and zero delay on step attacks. For ramp attacks, it requires 90 minutes of gradual accumulation before reaching the conservative safety threshold $h=18\text{ kW}$.
- **Isolation Forest:** By extracting the residual change rate $\Delta r(t)$, it catches the ramp attack in **18.0 minutes** (5x faster than CUSUM) with 95% recall, with a minor tradeoff of 0.33 false alarms per day.

---

## 5. Repository Structure

```
Secure Smart Charging/
├── pyproject.toml              # Build config and dependencies (Python 3.11+)
├── README.md                   # Complete academic documentation
├── src/
│   └── ev_sched/
│       ├── __init__.py         # Package exports
│       ├── models.py           # EVSession, Feeder, Tariff dataclasses with units
│       ├── hal.py              # Hardware Abstraction Layer (MeterInterface, EVSEInterface)
│       ├── simulator.py        # Synthetic session generator & 15-min simulator
│       ├── scheduler.py        # CVXPY optimization & receding-horizon MPC
│       ├── attacks.py          # FDI attack wrappers (Bias, Ramp, Replay)
│       ├── detector.py         # CUSUM & Isolation Forest cyber detectors
│       └── metrics.py          # Overload minutes, costs, unmet energy, FAR, delay
├── experiments/
│   ├── baseline_config.yaml    # Simulation parameters (Poisson arrival, dwell, tariff)
│   ├── run_baseline.py         # Baseline simulation script
│   ├── compare_baseline_vs_mpc.py # 20-seed Baseline vs MPC study
│   └── run_attack_and_detection_experiments.py # Tasks 11 & 14 execution script
├── tests/
│   ├── test_skeleton.py        # Module import tests
│   ├── test_models.py          # Dataclass validation & unit tests
│   ├── test_hal.py             # HAL contract verification
│   ├── test_simulator.py       # Deterministic generator & session tests
│   ├── test_baseline.py        # Energy & power boundary invariants
│   ├── test_scheduler.py       # CVXPY feasibility, status check & safe fallback
│   ├── test_attacks.py         # Attack interface wrapping & offset math
│   ├── test_detector.py        # CUSUM & Isolation Forest unit tests
│   └── test_metrics.py         # Metric calculation checks
└── results/                    # Git-tracked reproducible figures and CSVs
    ├── baseline_load.png       # Baseline uncontrolled feeder load vs transformer limit
    ├── baseline_vs_mpc.csv     # 20-seed comparison table
    ├── baseline_vs_mpc.png     # Load profiles & cost distribution plots
    ├── attack_impact.csv       # 80-run attack impact dataset
    ├── attack_impact.png       # True feeder load time-series under cyber attacks
    └── detection_metrics.csv   # Precision, Recall, FAR, Delay comparison table
```

---

## 6. How to Run & Reproduce

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Execute Pytest Suite (32 Tests)
```bash
PYTHONPATH=src:. pytest -v
```

### 3. Reproduce Baseline vs. MPC Experiment (Task 8)
```bash
PYTHONPATH=src:. python experiments/compare_baseline_vs_mpc.py
```
Outputs `results/baseline_vs_mpc.csv` and `results/baseline_vs_mpc.png`.

### 4. Reproduce Cyber Attack & Detector Evaluation (Tasks 11 & 14)
```bash
PYTHONPATH=src:. python experiments/run_attack_and_detection_experiments.py
```
Simulates 80 MPC runs across 20 seeds and outputs:
- `results/attack_impact.csv`
- `results/attack_impact.png`
- `results/detection_metrics.csv`

---

## 7. Assumptions & Engineering Decisions

1. **Active Power Flow Focus:** Feeder capacity constraints are evaluated using real active power ($P$ in kW). Power factor is assumed near-unity at EVSE inverter stages; reactive power and bus voltages can be integrated in future work via Second-Order Cone Programming (SOCP).
2. **Submeter Asymmetry:** Individual EVSE charging posts are located on-premises and communicate over authenticated TLS/OCPP links with local metering chips, whereas substation feeder meters communicate over utility WAN/SCADA links susceptible to man-in-the-middle FDI tampering.
3. **CUSUM Parameter Selection:** The slack parameter $\nu = 1.5\text{ kW}$ exceeds $3\sigma$ of measurement noise to guarantee negative drift under $H_0$. The threshold $h = 18.0\text{ kW}$ was tuned using Average Run Length theory to ensure zero false alarms during normal operations.
4. **Isolation Forest Features:** Providing the first difference $\Delta r(t) = r(t) - r(t-1)$ alongside $r(t)$ enables the tree partitioner to isolate ramping anomalies 5x faster than cumulative residual sums.
