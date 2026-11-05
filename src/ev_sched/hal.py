"""Hardware Abstraction Layer (HAL) for EV Charging and Grid Telemetry.

In physical deployments:
- MeterInterface maps to Modbus / DNP3 telemetry from substation RTUs or smart meters.
- EVSEInterface maps to OCPP 1.6J/2.0.1 (e.g. SetChargingProfile, MeterValues).

This layer guarantees that control and detection algorithms interact strictly
with standardized abstract interfaces, enabling drop-in physical deployment.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class MeterInterface(ABC):
    """Abstract interface for reading feeder / substation power meters."""

    @abstractmethod
    def read_feeder_kw(self, t_min: int) -> float:
        """[MEASURED] Read active power flow on the substation feeder.

        Parameters
        ----------
        t_min : int
            Elapsed time from simulation start in minutes [minutes].

        Returns
        -------
        float
            Measured active power in kilowatts [kW]. Subject to cyber tampering
            if attacked.
        """
        raise NotImplementedError


class EVSEInterface(ABC):
    """Abstract interface for EVSE (Electric Vehicle Supply Equipment) telemetry & control."""

    @abstractmethod
    def set_setpoint_kw(self, ev_id: str, power_kw: float) -> None:
        """Dispatch charging active power command to an individual EV charger.

        Parameters
        ----------
        ev_id : str
            Unique vehicle / session identifier.
        power_kw : float
            Commanded active power setpoint in kilowatts [kW]. Must be non-negative.
        """
        raise NotImplementedError

    @abstractmethod
    def read_session_kw(self, ev_id: str) -> float:
        """[MEASURED] Read the actual delivered active power at the EVSE submeter.

        Parameters
        ----------
        ev_id : str
            Unique vehicle / session identifier.

        Returns
        -------
        float
            Submetered active power delivered to the vehicle in kilowatts [kW].
        """
        raise NotImplementedError


class SimMeter(MeterInterface):
    """Simulator wrapper implementing MeterInterface for digital simulations.

    Parameters
    ----------
    simulator : Any
        Underlying grid and EV simulation instance.
    noise_std_kw : float, default 0.0
        Standard deviation of Gaussian sensor measurement noise in kilowatts [kW].
    """

    def __init__(self, simulator: Any, noise_std_kw: float = 0.0) -> None:
        self._sim = simulator
        self.noise_std_kw = noise_std_kw

    def read_feeder_kw(self, t_min: int) -> float:
        """[MEASURED] Return measured active feeder power from simulator."""
        raw_kw = self._sim.get_feeder_power_kw(t_min)
        if self.noise_std_kw > 0.0 and hasattr(self._sim, "rng"):
            noise = float(self._sim.rng.normal(0.0, self.noise_std_kw))
            return max(0.0, raw_kw + noise)
        return float(raw_kw)


class SimEVSE(EVSEInterface):
    """Simulator wrapper implementing EVSEInterface for digital simulations.

    Parameters
    ----------
    simulator : Any
        Underlying grid and EV simulation instance.
    """

    def __init__(self, simulator: Any) -> None:
        self._sim = simulator

    def set_setpoint_kw(self, ev_id: str, power_kw: float) -> None:
        """Dispatch setpoint to simulator."""
        self._sim.set_ev_setpoint_kw(ev_id, max(0.0, float(power_kw)))

    def read_session_kw(self, ev_id: str) -> float:
        """[MEASURED] Read active charging power of session from simulator."""
        return float(self._sim.get_session_power_kw(ev_id))
