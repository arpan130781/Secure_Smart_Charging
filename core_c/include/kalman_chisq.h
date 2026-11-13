/**
 * @file kalman_chisq.h
 * @brief Discrete-time Kalman Filter State Estimator and Chi-Square Anomaly Detector.
 *
 * Implements classical power system state estimation and chi-square residual
 * testing (replacing black-box ML like Isolation Forest) with O(1) memory and
 * deterministic execution time.
 */

#ifndef SEC_KALMAN_CHISQ_H
#define SEC_KALMAN_CHISQ_H

#include "sec_charge_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double x_hat;           /**< Estimated true feeder state [kW] */
    double P_cov;           /**< Estimation error covariance */
    double Q_process_noise; /**< Process noise variance Q */
    double R_meas_noise;    /**< Measurement noise variance R */
    double chisq_threshold; /**< Critical Chi-Square threshold (e.g. 9.0 for 3-sigma) */
    double ewma_chisq;      /**< Exponentially smoothed Chi-Square metric */
    double ewma_alpha;      /**< Smoothing factor in (0, 1] */
    bool initialized;       /**< Filter initialization flag */
} sec_kalman_chisq_t;

/**
 * @brief Initialize the Kalman Filter and Chi-Square anomaly detector.
 *
 * @param kf Pointer to detector state
 * @param q_process Process noise variance Q (default ~ 0.5)
 * @param r_meas Measurement noise variance R (default ~ 0.09 for sigma=0.3 kW)
 * @param chisq_threshold Chi-Square alert threshold (e.g. 9.0 for 3-sigma)
 * @param ewma_alpha Smoothing parameter (e.g. 0.3)
 * @return 0 on success, negative on error
 */
int sec_kalman_chisq_init(
    sec_kalman_chisq_t *kf,
    double q_process,
    double r_meas,
    double chisq_threshold,
    double ewma_alpha
);

/**
 * @brief Reset Kalman filter state.
 */
int sec_kalman_chisq_reset(sec_kalman_chisq_t *kf);

/**
 * @brief Ingest telemetry, execute Kalman update, and compute Chi-Square anomaly distance.
 *
 * @param kf Pointer to filter state
 * @param reported_feeder_kw Measured feeder load [kW]
 * @param expected_feeder_kw Model-predicted feeder load [kW]
 * @param alert Output alert structure
 * @return 0 on success, negative on error
 */
int sec_kalman_chisq_update(
    sec_kalman_chisq_t *kf,
    double reported_feeder_kw,
    double expected_feeder_kw,
    sec_detector_alert_t *alert
);

#ifdef __cplusplus
}
#endif

#endif /* SEC_KALMAN_CHISQ_H */
