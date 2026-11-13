"""False Data Injection Detectors for Grid Feeder Meters.

Includes:
1. Detector 1: Energy-balance residual check with Cumulative Sum (CUSUM) statistic.
2. Detector 2: Machine Learning Isolation Forest trained exclusively on attack-free runs.
"""

from dataclasses import dataclass

import numpy as np

try:
    from sklearn.ensemble import IsolationForest
except ImportError:
    IsolationForest = None  # type: ignore


@dataclass(frozen=True)
class DetectorAlert:
    """Telemetry diagnostic returned at each step by a detector.

    Parameters
    ----------
    is_alarm : bool
        True if the detector classifies the current time step as anomalous / under attack.
    residual_kw : float
        Innovation residual r(t) = reported_kw - expected_kw in kilowatts [kW].
    statistic_value : float
        Internal decision metric (CUSUM accumulator value [kW] or Isolation Forest score).
    """

    is_alarm: bool
    residual_kw: float
    statistic_value: float


class CUSUMDetector:
    """Detector 1: Physics-informed energy-balance residual check with CUSUM statistic.

    Physical Principle & Residual Computation:
    -----------------------------------------
    The central charging scheduler commands known charging power setpoints sum(P_cmd)
    and holds a base load forecast P_base. Under nominal physical conservation of energy:
        expected_feeder_kw = P_base + sum(P_cmd)
        residual_kw = reported_feeder_kw - expected_feeder_kw

    Under normal attack-free conditions, residual_kw represents zero-mean sensor noise:
        r(t) ~ Normal(0, sigma^2), where sigma approx 0.3 - 0.5 kW.

    Under an under-reporting cyber attack (reported = true - delta):
        r(t) approx -delta + noise < 0.

    CUSUM Formulation & Threshold Justification:
    --------------------------------------------
    A one-sided downward Page's CUSUM statistic tracks persistent negative residual drifts:
        S^-(0) = 0
        S^-(t) = max(0, S^-(t-1) - r(t) - nu)

    Threshold Choice:
    - Slack parameter (nu): Set to 1.5 kW (approx 3 * sigma). This guarantees that
      under H_0 (no attack), E[-r(t) - nu] = -nu < 0, ensuring the accumulator
      decays rapidly to zero and prevents false alarm accumulation.
    - Threshold (h): Set to 18.0 kW. By Page's Average Run Length (ARL) theory,
      the mean time between false alarms is:
          ARL_0 approx (exp(2 * nu * h / sigma^2) - 1) / (2 * (nu / sigma)^2)
      With nu = 1.5 kW and h = 18.0 kW, ARL_0 exceeds 10^5 steps (negligible false
      alarm rate per day), while detecting a 35 kW bias within 1 time step and
      detecting ramp attacks before transformer damage occurs.

    Parameters
    ----------
    threshold : float, default 18.0
        Decision boundary threshold h in kilowatts [kW].
    slack_kw : float, default 1.5
        Allowance parameter nu compensating for nominal measurement noise [kW].
    instant_threshold_kw : float, default 20.0
        Immediate alarm threshold on |residual| for large step attacks [kW].
    """

    def __init__(
        self,
        threshold: float = 18.0,
        slack_kw: float = 1.5,
        instant_threshold_kw: float = 20.0,
    ) -> None:
        self.threshold = threshold
        self.slack_kw = slack_kw
        self.instant_threshold_kw = instant_threshold_kw
        self.cusum_neg: float = 0.0

    def reset(self) -> None:
        """Reset internal accumulator for a new operational run."""
        self.cusum_neg = 0.0

    def update(
        self,
        reported_feeder_kw: float,
        expected_feeder_kw: float,
    ) -> DetectorAlert:
        """Evaluate meter telemetry at time step t.

        Parameters
        ----------
        reported_feeder_kw : float
            [MEASURED] Telemetry returned by the MeterInterface in kilowatts [kW].
        expected_feeder_kw : float
            [EXPECTED] Model-predicted power (base load forecast + sum(setpoints)) [kW].

        Returns
        -------
        DetectorAlert
            Alarm flag, residual [kW], and CUSUM statistic [kW].
        """
        residual = reported_feeder_kw - expected_feeder_kw

        # Downward CUSUM: detects negative residual (under-reporting attack)
        self.cusum_neg = max(0.0, self.cusum_neg - residual - self.slack_kw)

        # Trigger alarm if CUSUM boundary crossed or instant magnitude exceeds threshold
        is_alarm = bool(
            (self.cusum_neg >= self.threshold) or (residual <= -self.instant_threshold_kw)
        )

        stat = float(self.cusum_neg)
        if is_alarm:
            # Reset accumulator upon alarm to track distinct alarm events
            self.cusum_neg = 0.0

        return DetectorAlert(
            is_alarm=is_alarm,
            residual_kw=float(residual),
            statistic_value=stat,
        )


class KalmanChiSquareDetector:
    """Detector 2 (Classical Control): Discrete-time Kalman Filter State Estimator & Chi-Square Detector.

    Replaces black-box Machine Learning with a deterministic, physically grounded dynamic state observer.
    Ingests active power measurements, predicts physical state evolution, and tests the normalized
    innovation residual against a Chi-Square (chi^2) distribution. Executes in O(1) time and memory,
    ideal for microcontrollers and embedded Energy Management Systems.
    """

    def __init__(
        self,
        q_process: float = 0.5,
        r_meas: float = 0.09,
        chisq_threshold: float = 9.0,
        ewma_alpha: float = 0.3,
    ) -> None:
        self.q_process = q_process
        self.r_meas = r_meas
        self.chisq_threshold = chisq_threshold
        self.ewma_alpha = ewma_alpha
        self.x_hat: float = 0.0
        self.p_cov: float = 1.0
        self.ewma_chisq: float = 0.0
        self.initialized: bool = False
        self.prev_residual_kw: float = 0.0

    def reset(self) -> None:
        """Reset internal filter state."""
        self.x_hat = 0.0
        self.p_cov = 1.0
        self.ewma_chisq = 0.0
        self.initialized = False
        self.prev_residual_kw = 0.0

    def fit_from_attack_free_runs(self, training_feature_matrix: np.ndarray) -> None:
        """Calibrate measurement noise variance R from attack-free baseline telemetry."""
        if len(training_feature_matrix) > 0:
            residuals = training_feature_matrix[:, 0]
            var = float(np.var(residuals))
            if var > 1e-4:
                self.r_meas = var

    def extract_features(
        self,
        reported_feeder_kw: float,
        expected_feeder_kw: float,
        prev_residual_kw: float | None = None,
    ) -> np.ndarray:
        """Construct feature vector for step t."""
        residual = reported_feeder_kw - expected_feeder_kw
        prev_r = self.prev_residual_kw if prev_residual_kw is None else prev_residual_kw
        delta_r = residual - prev_r
        ratio = reported_feeder_kw / max(1.0, expected_feeder_kw)
        return np.array([residual, delta_r, ratio], dtype=float)

    def update(
        self,
        reported_feeder_kw: float,
        expected_feeder_kw: float,
    ) -> DetectorAlert:
        """Evaluate meter telemetry using discrete-time Kalman filter and Chi-Square anomaly metric."""
        if not self.initialized:
            self.x_hat = expected_feeder_kw
            self.p_cov = self.q_process
            self.initialized = True

        # 1. Time update (Predict)
        x_prior = expected_feeder_kw
        p_prior = self.p_cov + self.q_process

        # 2. Measurement innovation: FDI attacks hide load (under-report)
        innovation = expected_feeder_kw - reported_feeder_kw
        innovation_cov = p_prior + self.r_meas

        # 3. Measurement update (Correct)
        k_gain = p_prior / innovation_cov
        self.x_hat = x_prior - (k_gain * innovation)
        self.p_cov = (1.0 - k_gain) * p_prior

        # 4. Chi-Square distance: d^2 = innovation^2 / S
        chisq_dist = (innovation * innovation) / innovation_cov
        self.ewma_chisq = (self.ewma_alpha * chisq_dist) + ((1.0 - self.ewma_alpha) * self.ewma_chisq)

        # Anomaly decision rule
        is_alarm = bool((chisq_dist > self.chisq_threshold) or (self.ewma_chisq > 0.6 * self.chisq_threshold))
        self.prev_residual_kw = float(innovation)

        return DetectorAlert(
            is_alarm=is_alarm,
            residual_kw=float(innovation),
            statistic_value=float(chisq_dist),
        )


class IsolationForestDetector:
    """Detector 2: Machine Learning Isolation Forest anomaly detector.

    Features Extracted at Step t:
    -----------------------------
    1. residual_kw : r(t) = reported_feeder_kw - expected_feeder_kw [kW]
    2. residual_change_kw : Delta r(t) = r(t) - r(t-1) [kW / 15-min interval]
    3. ratio : reported_feeder_kw / (expected_feeder_kw + 1e-3) [dimensionless]

    Training Protocol:
    ------------------
    Trained strictly on attack-free operational telemetry across multiple normal runs.
    Learns the baseline distribution of legitimate forecast errors, EV tracking delays,
    and sensor noise.

    Parameters
    ----------
    contamination : float, default 0.01
        Expected proportion of outliers in normal baseline dataset.
    random_state : int, default 42
        Seed for reproducible random partitioning trees.
    """

    def __init__(self, contamination: float = 0.01, random_state: int = 42) -> None:
        self.contamination = contamination
        self.random_state = random_state
        if IsolationForest is not None:
            self.model = IsolationForest(
                contamination=contamination,
                random_state=random_state,
                n_estimators=100,
            )
        else:
            self.model = None
            self._kalman_fallback = KalmanChiSquareDetector()
        self.is_fitted: bool = False
        self.prev_residual_kw: float = 0.0

    def extract_features(
        self,
        reported_feeder_kw: float,
        expected_feeder_kw: float,
        prev_residual_kw: float | None = None,
    ) -> np.ndarray:
        """Construct feature vector for step t."""
        residual = reported_feeder_kw - expected_feeder_kw
        prev_r = self.prev_residual_kw if prev_residual_kw is None else prev_residual_kw
        delta_r = residual - prev_r
        ratio = reported_feeder_kw / max(1.0, expected_feeder_kw)
        return np.array([residual, delta_r, ratio], dtype=float)

    def fit_from_attack_free_runs(self, training_feature_matrix: np.ndarray) -> None:
        """Fit Isolation Forest using clean, unattacked operational data.

        Parameters
        ----------
        training_feature_matrix : np.ndarray
            Matrix of shape (M, 3) where each row is [residual, delta_residual, ratio].
        """
        if self.model is not None:
            self.model.fit(training_feature_matrix)
        elif hasattr(self, "_kalman_fallback"):
            self._kalman_fallback.fit_from_attack_free_runs(training_feature_matrix)
        self.is_fitted = True

    def reset(self) -> None:
        """Reset step-to-step memory."""
        self.prev_residual_kw = 0.0
        if hasattr(self, "_kalman_fallback"):
            self._kalman_fallback.reset()

    def update(
        self,
        reported_feeder_kw: float,
        expected_feeder_kw: float,
    ) -> DetectorAlert:
        """Evaluate meter telemetry using trained Isolation Forest.

        Parameters
        ----------
        reported_feeder_kw : float
            [MEASURED] Meter telemetry in kilowatts [kW].
        expected_feeder_kw : float
            [EXPECTED] Expected power (base load + sum(cmd)) in kilowatts [kW].

        Returns
        -------
        DetectorAlert
            Alarm flag, residual [kW], and decision function anomaly score.
        """
        if self.model is None and hasattr(self, "_kalman_fallback"):
            return self._kalman_fallback.update(reported_feeder_kw, expected_feeder_kw)

        feat = self.extract_features(reported_feeder_kw, expected_feeder_kw).reshape(1, -1)
        residual = float(feat[0, 0])
        self.prev_residual_kw = residual

        if not self.is_fitted:
            # If not yet trained, default to unalarmed
            return DetectorAlert(is_alarm=False, residual_kw=residual, statistic_value=0.0)

        # Isolation forest: -1 indicates anomaly, 1 indicates inlier
        pred = self.model.predict(feat)[0]
        score = float(self.model.decision_function(feat)[0])
        is_alarm = bool(pred == -1)

        return DetectorAlert(
            is_alarm=is_alarm,
            residual_kw=residual,
            statistic_value=score,
        )
