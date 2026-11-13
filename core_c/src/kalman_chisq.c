/**
 * @file kalman_chisq.c
 * @brief Implementation of Kalman Filter and Chi-Square residual detector in ANSI C99.
 */

#include "kalman_chisq.h"
#include <math.h>

int sec_kalman_chisq_init(
    sec_kalman_chisq_t *kf,
    double q_process,
    double r_meas,
    double chisq_threshold,
    double ewma_alpha
) {
    if (kf == NULL) {
        return SEC_CHARGE_ERR_NULL_PTR;
    }
    if (q_process <= 0.0 || r_meas <= 0.0 || chisq_threshold <= 0.0) {
        return SEC_CHARGE_ERR_PARAM;
    }
    if (ewma_alpha <= 0.0 || ewma_alpha > 1.0) {
        return SEC_CHARGE_ERR_PARAM;
    }

    kf->x_hat = 0.0;
    kf->P_cov = 1.0;
    kf->Q_process_noise = q_process;
    kf->R_meas_noise = r_meas;
    kf->chisq_threshold = chisq_threshold;
    kf->ewma_chisq = 0.0;
    kf->ewma_alpha = ewma_alpha;
    kf->initialized = false;

    return SEC_CHARGE_OK;
}

int sec_kalman_chisq_reset(sec_kalman_chisq_t *kf) {
    if (kf == NULL) {
        return SEC_CHARGE_ERR_NULL_PTR;
    }
    kf->x_hat = 0.0;
    kf->P_cov = 1.0;
    kf->ewma_chisq = 0.0;
    kf->initialized = false;
    return SEC_CHARGE_OK;
}

int sec_kalman_chisq_update(
    sec_kalman_chisq_t *kf,
    double reported_feeder_kw,
    double expected_feeder_kw,
    sec_detector_alert_t *alert
) {
    if (kf == NULL || alert == NULL) {
        return SEC_CHARGE_ERR_NULL_PTR;
    }

    if (!kf->initialized) {
        kf->x_hat = expected_feeder_kw;
        kf->P_cov = kf->Q_process_noise;
        kf->initialized = true;
    }

    /* 1. Time Update (Predict): prior state tracks expected physical load changes */
    double x_prior = expected_feeder_kw;
    double p_prior = kf->P_cov + kf->Q_process_noise;

    /* 2. Innovation: residual between measurement and prior prediction */
    /* FDI attacks hide load (under-report), so expected > reported is anomalous */
    double innovation = expected_feeder_kw - reported_feeder_kw;
    double innovation_cov = p_prior + kf->R_meas_noise;

    /* 3. Measurement Update: Kalman gain and posterior state */
    double k_gain = p_prior / innovation_cov;
    kf->x_hat = x_prior - (k_gain * innovation);
    kf->P_cov = (1.0 - k_gain) * p_prior;

    /* 4. Chi-Square normalized residual distance: d^2 = nu^2 / S */
    double chisq_dist = (innovation * innovation) / innovation_cov;

    /* Exponential smoothing of Chi-Square statistic */
    kf->ewma_chisq = (kf->ewma_alpha * chisq_dist) + ((1.0 - kf->ewma_alpha) * kf->ewma_chisq);

    /* Anomaly decision: instantaneous or smoothed threshold */
    bool is_alarm = (chisq_dist > kf->chisq_threshold) || (kf->ewma_chisq > (0.6 * kf->chisq_threshold));

    alert->is_alarm = is_alarm;
    alert->residual_kw = innovation;
    alert->test_statistic = chisq_dist;

    return SEC_CHARGE_OK;
}
