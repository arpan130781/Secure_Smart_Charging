# Secure Smart EV Charging: Mitigating False Data Injection on Feeder Meters

A modular, reproducible cyber-physical research framework studying how **False Data Injection (FDI)** attacks on substation feeder meter readings degrade **Model Predictive Control (MPC)** charging schedulers, and evaluating a **lightweight, physics-informed detector** that restores grid security and charging performance.

Designed for academic presentation to a university professor working on EV charging security and power grid analytics.

---

## 1. Executive Summary & Problem Motivation

In modern distribution networks, electric vehicle supply equipment (EVSE) clusters are managed by central Energy Management Systems (EMS). To avoid overloading distribution transformers and feeder cables, the EMS employs **Model Predictive Control (MPC)** to schedule EV charging within the available feeder capacity:

$$\sum_{i \in \text{EVs}} P_i(t) \le P_{\text{feeder}}^{\max} - \hat{P}_{\text{base}}(t)$$

Where:
- $P_{\text{feeder}}^{\max}$ is the feeder capacity rating [kW].
- $\hat{P}_{\text{base}}(t)$ is the estimated unmanaged background grid load [kW].
- $P_i(t)$ is the commanded charging power for vehicle $i$ [kW].

### The Cyber Vulnerability
Substation feeder meters communicate over SCADA / AMI links (e.g., Modbus/TCP, DNP3, or cellular). If an adversary tampers with this telemetry:
1. **Under-reporting Attack (Feeder Overload):** Falsely reporting low feeder load induces the MPC scheduler to perceive phantom headroom. The scheduler commands peak EV charging, driving physical power beyond transformer thermal ratings ($P_{\text{feeder}}(t) \gg P_{\text{feeder}}^{\max}$), causing transformer degradation, accelerated thermal aging, or protection trips.
2. **Over-reporting Attack (Denial-of-Service):** Falsely reporting high feeder load causes unnecessary curtailment, leading to severe departure energy shortfalls.

### The Solution
A **lightweight, physics-informed residual and CUSUM detector** that leverages independent submetered EVSE measurements to verify the feeder balance equation. Upon detecting an anomaly, the detector replaces the compromised meter reading with a safe, model-based reconstructed power estimate, completely restoring transformer safety without degrading EV Quality of Service (QoS).

---

## 2. Cyber-Physical Architecture & Hardware Abstraction Layer (HAL)

A core engineering principle of this framework is strict **separation of ground-truth physics from controller logic via a Hardware Abstraction Layer (HAL)**. The MPC scheduler and detector *never* access the simulator directly.

```
+-------------------------------------------------------------------+
|                        PHYSICAL WORLD                             |
|  - Substation Transformer (P_feeder_max = 100 kW)                |
|  - Diurnal Non-EV Base Load P_base(t) [SIMULATED]                 |
|  - Battery Chemical Integration E_i(t+1) [SIMULATED]              |
|  - Feeder Physical Power Flow P_feeder(t) = P_base + sum(P_i)     |
+---------------------------------+---------------------------------+
                                  |
               +------------------+------------------+
               |                                     |
               v                                     v
+-----------------------------+       +-----------------------------+
|    hal.MeterInterface       |       |     hal.EVSEInterface       |
|  read_feeder_power_kw()     |       |  get_connected_evs()        |
|  get_feeder_limit_kw()      |       |  set_charging_power_kw()    |
|                             |       |  get_total_measured_ev_kw() |
|  [MEASURED Telemetry]       |       |  [MEASURED Telemetry]       |
+--------------+--------------+       +--------------+--------------+
               |                                     |
               |       +---------------------+       |
               +------>| Lightweight Detector |<------+
               |       | (Physics Invariant) |
               |       +----------+----------+
               |                  | (restored reading if attacked)
               v                  v
+-------------------------------------------------------------------+
|                   CONTROLLER / SCHEDULER                          |
|  - Receding Horizon MPC (Linear Program via HiGHS solver)         |
|  - Allocates EV charging setpoints P_i(t)                         |
|  - Operates purely via HAL interfaces                             |
+-------------------------------------------------------------------+
```

### Simulated vs Measured Distinctions

| Variable | Class | Label | Physical Meaning & Units |
|---|---|---|---|
| `sim_base_load_kw` | `GridEVSimulator` | **[SIMULATED]** | True unmanaged background building/residential grid load [kW] |
| `sim_feeder_total_power_kw` | `GridEVSimulator` | **[SIMULATED]** | True total active power through substation conductor [kW] |
| `sim_feeder_overload_kw` | `GridEVSimulator` | **[SIMULATED]** | Ground-truth transformer violation $\max(0, P_{\text{feeder}} - P_{\text{feeder}}^{\max})$ [kW] |
| `energy_kwh` | `EVState` | **[SIMULATED]** | True battery chemical stored energy [kWh] |
| `meas_feeder_power_kw` | `MeterInterface` | **[MEASURED]** | Telemetry reported by feeder head meter, prone to noise & FDI [kW] |
| `measured_power_kw` | `EVSEInterface` | **[MEASURED]** | Submetered power delivered at individual charging stall [kW] |
| `current_energy_kwh` | `EVTelemetry` | **[MEASURED]** | Battery state of charge reported over EVSE/BMS protocol [kWh] |
| `restored_feeder_power_kw` | `DetectionResult` | **[SYNTHESIZED]** | Physics-reconstructed safe feeder reading during attack [kW] |

---

## 3. Mathematical Formulations

### 3.1 Model Predictive Control (MPC) Scheduler
At each time step $k$, with lookahead horizon $H$ (e.g., $H=12$ steps = 3 hours), the scheduler solves a Linear Program (LP):

$$\min_{\mathbf{P}, \mathbf{s}} \sum_{i=1}^{N} c_{\text{deficit}} \cdot s_i + \sum_{i=1}^{N} \sum_{h=0}^{H-1} c_{\text{energy}} \cdot P_{i, h} \cdot \Delta t_{\text{hours}}$$

Subject to:
1. **Transformer Capacity Constraint:**
   $$\sum_{i=1}^{N} P_{i, h} \le \max\left(0, P_{\text{feeder}}^{\max} - \hat{P}_{\text{base}}(k+h)\right), \quad \forall h \in [0, H-1]$$
2. **EV Power Saturation:**
   $$0 \le P_{i, h} \le P_i^{\max}, \quad \forall i, h$$
3. **Energy Target Fulfillment:**
   $$E_i(k) + \sum_{h=0}^{t_i^{\text{dep}}-k-1} \eta_i \cdot P_{i, h} \cdot \Delta t_{\text{hours}} + s_i \ge E_i^{\text{target}}, \quad s_i \ge 0$$
4. **Battery Energy Upper Bound:**
   $$E_i(k) + \sum_{\tau=0}^{h} \eta_i \cdot P_{i, \tau} \cdot \Delta t_{\text{hours}} \le E_i^{\text{cap}}, \quad \forall h$$

The optimal setpoint for the current step $P_i^*(k) = P_{i, 0}^*$ is dispatched to `EVSEInterface.set_charging_power_kw()`.

### 3.2 Physics-Informed Lightweight Detector & Restoration
The detector exploits the conservation of active power at the charging facility bus:
$$P_{\text{base}}^{\text{inferred}}(k) = y_{\text{meter}}(k) - \sum_{i} P_{i}^{\text{meas}}(k)$$

1. **Innovation Residual:**
   $$r(k) = P_{\text{base}}^{\text{inferred}}(k) - \hat{P}_{\text{base}}^{\text{prior}}(k)$$
2. **CUSUM Accumulators (detects gradual drift & ramp attacks):**
   $$S^+(k) = \max\left(0, S^+(k-1) + r(k) - \mu_{\text{slack}}\right)$$
   $$S^-(k) = \max\left(0, S^-(k-1) - r(k) - \mu_{\text{slack}}\right)$$
3. **Alarm Condition:**
   $$\text{Alarm} = (P_{\text{base}}^{\text{inferred}}(k) < 0) \lor (|r(k)| > \tau_{\text{thresh}}) \lor (\max(S^+, S^-) > h_{\text{cusum}})$$
4. **Trustworthy State Restoration:**
   $$\hat{P}_{\text{feeder}}^{\text{restored}}(k) = \hat{P}_{\text{base}}^{\text{prior}}(k) + \gamma_{\text{safety}} + \sum_i P_i^{\text{meas}}(k)$$

---

## 4. Experimental Results

Three benchmark cyber conditions were executed across a 24-hour simulation (96 time steps at 15-minute intervals, $P_{\text{feeder}}^{\max} = 100.0\text{ kW}$, 8 commercial EVs demanding 303.0 kWh total):

### Comparative Performance Table

| Scenario | Peak Feeder Power | Peak Overload | Overload Duration | Overload Energy | Delivered Energy | Unmet Energy | Fulfillment Rate | Detection TPR | Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Baseline (No Attack)** | 100.76 kW | 0.76 kW* | 45.0 min | 0.27 kWh | 303.00 kWh | 0.00 kWh | 100.0% | N/A | N/A |
| **2. Under Attack (Unmitigated)** | **117.51 kW** | **17.51 kW** | **75.0 min** | **12.32 kWh** | 303.00 kWh | 0.00 kWh | 100.0% | 0.0% | N/A |
| **3. Under Attack (Mitigated)** | **96.00 kW** | **0.00 kW** | **0.0 min** | **0.00 kWh** | 303.00 kWh | 0.00 kWh | 100.0% | **100.0%** | **0 min** |

*\* Minor noise peak of 0.76 kW (<0.8%) is due to zero-mean sensor measurement noise ($3\sigma = 0.9$ kW).*

### Key Findings
1. **Vulnerability of Standard MPC:** When the attacker injects an under-reporting bias of $-35\text{ kW}$ between 09:00 and 16:00, the MPC scheduler is misled into believing substantial headroom exists. It commands peak EV charging, pushing physical feeder load to **117.51 kW** (a **17.51% transformer overload**) for **75 consecutive minutes**, accumulating **12.32 kWh** of thermal overload.
2. **Zero-Latency Detection:** The physics-based detector identifies the attack immediately on step 36 ($t=0\text{ min}$ latency) with **100% True Positive Rate (TPR)** and **0% False Positive Rate (FPR)**.
3. **Complete Grid Protection:** With restoration active, peak physical feeder power is restricted to **96.0 kW**, completely eliminating transformer overload (**0.0 kW peak overload, 0.0 min duration, 0.0 kWh overload energy**), while maintaining **100% EV target energy delivery**.

---

## 5. Repository Structure

```
Secure Smart Charging/
├── pyproject.toml              # Build config and dependency definitions
├── .gitignore                  # Excludes venv, caches; tracks results/
├── README.md                   # Complete academic documentation
├── src/
│   └── ev_sched/
│       ├── __init__.py         # Package exports
│       ├── models.py           # Dataclasses with explicit units (kW, kWh, min)
│       ├── hal.py              # MeterInterface & EVSEInterface protocols
│       ├── simulator.py        # Physical environment & battery state integration
│       ├── scheduler.py        # Receding-horizon MPC linear program (HiGHS)
│       ├── attacks.py          # FDI attack models (bias, scaling, ramp)
│       ├── detector.py         # Physics-informed residual & CUSUM detector
│       └── metrics.py          # Grid safety, EV QoS, and detector metrics
├── experiments/
│   ├── config_feeder_overload_bias.yaml   # Reproducible 35 kW bias scenario
│   ├── config_feeder_overload_ramp.yaml   # Stealthy 2 kW/step ramp scenario
│   ├── config_dos_over_reporting.yaml     # Denial-of-service +35 kW scenario
│   └── run_experiments.py                # Main experiment execution script
├── tests/
│   ├── test_models_and_hal.py
│   ├── test_simulator.py
│   ├── test_scheduler.py
│   ├── test_attacks.py
│   ├── test_detector.py
│   └── test_end_to_end_experiment.py
└── results/                    # Git-tracked figures and CSV datasets
    ├── feeder_overload_bias_evaluation.png
    ├── feeder_overload_bias_metrics_comparison.csv
    ├── feeder_overload_bias_timeseries_mitigated.csv
    ├── feeder_overload_ramp_evaluation.png
    ├── feeder_overload_ramp_metrics_comparison.csv
    ├── feeder_overload_ramp_timeseries_mitigated.csv
    ├── dos_over_reporting_evaluation.png
    ├── dos_over_reporting_metrics_comparison.csv
    └── dos_over_reporting_timeseries_mitigated.csv
```

---

## 6. How to Run & Reproduce

### Prerequisites
- Python 3.11 or higher
- Git

### 1. Environment Setup
```bash
cd "Secure Smart Charging"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Run Test Suite
```bash
PYTHONPATH=src:. pytest -v
```
Output:
```
============================== 10 passed in 0.77s ==============================
```

### 3. Run Experiments
Execute the feeder overload bias experiment:
```bash
PYTHONPATH=src python experiments/run_experiments.py --config experiments/config_feeder_overload_bias.yaml
```

Execute the stealthy ramp attack experiment:
```bash
PYTHONPATH=src python experiments/run_experiments.py --config experiments/config_feeder_overload_ramp.yaml
```

Execute the Denial-of-Service over-reporting experiment:
```bash
PYTHONPATH=src python experiments/run_experiments.py --config experiments/config_dos_over_reporting.yaml
```

All results (plots and CSVs) are automatically written to `results/`.

---

## 7. Assumptions & Engineering Decisions

1. **Active Power Flow Focus:** In distribution grid EV charging studies, active power ($P$ [kW]) governs transformer thermal capacity and conductor ampacity. Reactive power ($Q$) and voltage drop can be integrated via second-order cone constraints (SOCP) in future work, but a Linear Program (LP) provides deterministic, global optimality solvable in milliseconds.
2. **Submeter Integrity:** We assume EVSE internal submeters communicate over authenticated TLS/OCPP channels. This asymmetry is standard in cyber-physical security: substation telemetry traversing long-distance utility backhauls is more susceptible to interception than on-premises EVSE submeters.
3. **Persistence and Diurnal Baseline:** Feeder baseline loads follow well-established diurnal cycles. The detector relies on this continuity; anomalous departures exceeding physical ramp limits are immediately flagged.
