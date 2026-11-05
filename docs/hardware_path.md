# Hardware Path: Transitioning from Simulation to Physical Deployment

This document details the engineering roadmap for deploying the **Secure Smart EV Charging** framework onto physical embedded hardware and standards-compliant EVSE equipment.

The modular Hardware Abstraction Layer (`hal.py`) was explicitly designed so that simulated classes (`SimMeter`, `SimEVSE`) can be swapped for physical drivers without modifying the optimization scheduler, bad-data detector, or cyber-resilience logic.

---

## 1. Physical Architecture Overview

```
+---------------------------------------------------------------------------------+
|                       ELECTRICAL DISTRIBUTION FEEDER                            |
|  - 400V 3-Phase / 230V Single-Phase Substation Transformer                      |
|  - Primary Current Transformers (CT) & Voltage Sensing                          |
+----------------------------------------+----------------------------------------+
                                         |
                       +-----------------+-----------------+
                       |                                   |
                       v                                   v
        +-----------------------------+     +-----------------------------+
        |   FEEDER METERING NODE      |     |  COMMERCIAL EV CHARGING HUB |
        |  - Microchip ATM90E26 /     |     |  - Commercial OCPP 1.6-J    |
        |    TI INA219                |     |    Charging Stations        |
        |  - Host: Raspberry Pi /     |     |  - In-stall MID submetering |
        |    ESP32 via SPI / I2C      |     |  - Type 2 / CCS2 Connectors |
        +--------------+--------------+     +--------------+--------------+
                       |                                   |
            Modbus/TCP | Cellular                          | OCPP 1.6-J / TLS
                       v                                   v
+---------------------------------------------------------------------------------+
|                    CENTRAL ENERGY MANAGEMENT SYSTEM (EMS)                       |
|                                                                                 |
|  +------------------------+                     +----------------------------+  |
|  | HardwareMeterDriver    |                     | OCPPEVSEGateway            |  |
|  | implements             |                     | implements                 |  |
|  | hal.MeterInterface     |                     | hal.EVSEInterface          |  |
|  +-----------+------------+                     +-------------+--------------+  |
|              |                                                |                 |
|              v                                                v                 |
|       +--------------------------------------------------------------+          |
|       |                     CYBER FAULT DETECTOR                     |          |
|       |  - Physics-Informed CUSUM & Isolation Forest Residual Checks |          |
|       +------------------------------+-------------------------------+          |
|                                      |                                          |
|                                      v                                          |
|       +--------------------------------------------------------------+          |
|       |                     MPC CHARGING CONTROLLER                  |          |
|       |  - CVXPY Quadratic Program (Clarabel / OSQP)                 |          |
|       |  - Dispatches Power Setpoints to EVSE Gateway                |          |
|       +--------------------------------------------------------------+          |
+---------------------------------------------------------------------------------+
```

---

## 2. Replacing `SimMeter` with Physical Power Measurement

### 2.1 Hardware Sensor Selection

| Component | Target Application | Bus Interface | Key Specifications |
|---|---|---|---|
| **Microchip ATM90E26** | **AC Feeder Transformer Monitoring** (Recommended for grid deployment) | SPI / UART | Single-phase energy metering AFE; dynamic range 7200:1; 0.1% accuracy; active/reactive/apparent power; RMS current & voltage; on-chip calibration registers. |
| **Texas Instruments INA219** | **DC Bus / Low-Voltage Testbed** (Benchtop prototyping) | I2C (up to 3.4 MHz) | High-side current & voltage sensor; 0–26V DC common-mode; 12-bit ADC; programmable calibration register directly reading current in mA and power in mW. |

For a 400V/230V commercial grid connection, **Microchip ATM90E26** (or three ATM90E26 ICs for polyphase / ATM90E36A for three-phase) is selected. It interfaces with split-core current transformers (e.g. 100A:50mA) clamped around feeder phase conductors and a step-down potential transformer for voltage sensing.

### 2.2 Host Controller Architecture
- **Host Platform:** Raspberry Pi 4 Model B (Linux CM4 industrial module) or ESP32-WROOM-32.
- **Physical Interface:** Linux SPI interface `/dev/spidev0.0` running at 200 kHz mode 1 (CPOL=0, CPHA=1) with dedicated hardware Chip Select (`GPIO 8`).
- **Data Acquisition Daemon:** A lightweight C or Python daemon reads the active power register at 1 Hz, applies calibration scaling, filters spikes, and publishes measurements over authenticated MQTT or Modbus/TCP to the EMS.

### 2.3 Concrete Interface Implementation
The hardware driver wraps the communication link and implements `hal.MeterInterface`:

```python
from ev_sched.hal import MeterInterface
import spidev
import time

class HardwareATM90E26Meter(MeterInterface):
    """Physical Feeder Meter driver reading Microchip ATM90E26 over SPI."""

    REG_P_MEAN = 0x4A      # Mean Active Power register
    REG_U_RMS  = 0x49      # RMS Voltage register
    REG_I_RMS  = 0x48      # RMS Current register
    POWER_SCALE = 0.0003125 # Calibration multiplier [kW per register LSB]

    def __init__(self, bus: int = 0, device: int = 0) -> None:
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = 200000
        self.spi.mode = 1

    def _read_register_16(self, reg_addr: int) -> int:
        # ATM90E26 SPI read frame: [ADDR | 0x80, 0x00, 0x00]
        cmd = [(reg_addr & 0x7F) | 0x80, 0x00, 0x00]
        resp = self.spi.xfer2(cmd)
        raw_val = (resp[1] << 8) | resp[2]
        return raw_val

    def read_feeder_kw(self, t_min: int) -> float:
        """[MEASURED] Return active feeder power in kilowatts [kW]."""
        try:
            raw_power = self._read_register_16(self.REG_P_MEAN)
            # Handle two's complement for signed real power
            if raw_power > 32767:
                raw_power -= 65536
            power_kw = abs(raw_power * self.POWER_SCALE)
            return float(power_kw)
        except Exception as err:
            # Safe failover on bus timeout: return conservative estimate
            return 120.0
```

---

## 3. Replacing `SimEVSE` with OCPP 1.6-J (JSON over WebSockets)

### 3.1 OCPP Protocol Role
In production, the EMS operates as an **OCPP Central System (CSMS)** communicating with physical EVSE charging stations over WebSockets with TLS encryption (`wss://`).

### 3.2 Concrete Protocol Mapping to `hal.EVSEInterface`

#### 1. Dispatching Power Setpoints (`set_setpoint_kw`)
When the MPC solver generates charging power $P_i(t)$ [kW], the gateway constructs an OCPP `SetChargingProfile.req` PDU:

```json
[
  2,
  "msg-unique-id-1001",
  "SetChargingProfile",
  {
    "connectorId": 1,
    "csChargingProfiles": {
      "chargingProfileId": 101,
      "stackLevel": 1,
      "chargingProfilePurpose": "TxProfile",
      "chargingProfileKind": "Absolute",
      "recurrencyKind": "Daily",
      "chargingSchedule": {
        "duration": 900,
        "chargingRateUnit": "W",
        "chargingSchedulePeriod": [
          {
            "startPeriod": 0,
            "limit": 11000.0
          }
        ]
      }
    }
  }
]
```
- **Unit Conversion:** Setpoint in kW is multiplied by $1000$ to conform to OCPP `limit` in Watts (or divided by nominal voltage 230V if `chargingRateUnit` is Amperes).
- **Control Duration:** 900 seconds (15-minute MPC slot).

#### 2. Reading Submetered Power (`read_session_kw`)
Physical EVSEs contain calibrated internal MID meters. The EVSE periodically sends `MeterValues.req` messages:

```json
[
  2,
  "msg-unique-id-1002",
  "MeterValues",
  {
    "connectorId": 1,
    "transactionId": 5401,
    "meterValue": [
      {
        "timestamp": "2026-11-05T09:15:00Z",
        "sampledValue": [
          {
            "value": "10842.0",
            "context": "Sample.Periodic",
            "format": "Raw",
            "measurand": "Power.Active.Import",
            "phase": "L1-L2-L3",
            "location": "Outlet",
            "unit": "W"
          }
        ]
      }
    ]
  }
]
```
The driver parses `sampledValue` where `measurand == "Power.Active.Import"`:
$$\text{power\_kw} = \frac{\text{float}(\text{value})}{1000.0} = 10.842\text{ kW}$$

---

## 4. Hardware Verification & Phased Test Plan

### Phase 1: Bus & Protocol Loopback Test (Benchtop)
- **Objective:** Verify communication drivers without high-voltage power.
- **Test Setup:**
  - Connect Raspberry Pi SPI port to an ATM90E26 evaluation module.
  - Send register read requests to configuration registers `0x00` (`SoftReset`) and `0x40` (`SysStatus`).
  - Connect a software mock EVSE (using the open-source `ocpp` Python library) over local WebSocket `ws://localhost:9000`.
  - Transmit `SetChargingProfile.req` and verify `SetChargingProfile.conf` is received within $200\text{ ms}$.

### Phase 2: Electrical Calibration & Signal Integrity
- **Objective:** Ground-truth measurement calibration against precision laboratory standards.
- **Test Setup:**
  - Connect ATM90E26 current inputs to a programmable AC current calibrator (0–50A RMS, $50\text{ Hz}$).
  - Measure register counts across 10 stepped current levels (5A, 10A, ..., 50A).
  - Compute gain coefficient $K_I$ and phase angle compensation $\theta_{\text{comp}}$ using linear least-squares regression.
  - Verify that residual noise standard deviation under steady-state load satisfies $\sigma \le 0.3\text{ kW}$, directly validating the slack parameter $\nu = 1.5\text{ kW}$ chosen for the CUSUM detector.

### Phase 3: Cyber-Fault & Tampering Injection Tests
- **Objective:** Validate bad-data detection and controller safe-mode failover under physical hardware conditions.
- **Test Setup:**
  - Insert a network proxy (mitmproxy / eBPF filter) on the feeder meter telemetry link.
  - Inject $-35.0\text{ kW}$ bias and ramp drifts into reported Modbus/TCP packets.
  - **Pass Criteria:**
    1. CUSUM detector flags alarm within $\le 1$ time step of bias injection.
    2. EMS seamlessly switches to model expectation $\hat{P}_{\text{base}}$ without solver failure.
    3. Derated setpoints are dispatched via OCPP `SetChargingProfile` within $500\text{ ms}$, maintaining transformer physical load below $120.0\text{ kW}$.
