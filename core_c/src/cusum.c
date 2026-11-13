/**
 * @file cusum.c
 * @brief Implementation of CUSUM sequential test in ANSI C99.
 */

#include "cusum.h"
#include <math.h>

int sec_cusum_init(sec_cusum_t *det, double threshold_kw, double slack_kw, double instant_threshold_kw) {
    if (det == NULL) {
        return SEC_CHARGE_ERR_NULL_PTR;
    }
    if (threshold_kw <= 0.0 || slack_kw < 0.0 || instant_threshold_kw <= 0.0) {
        return SEC_CHARGE_ERR_PARAM;
    }
    det->threshold_kw = threshold_kw;
    det->slack_kw = slack_kw;
    det->instant_threshold_kw = instant_threshold_kw;
    det->statistic = 0.0;
    return SEC_CHARGE_OK;
}

int sec_cusum_reset(sec_cusum_t *det) {
    if (det == NULL) {
        return SEC_CHARGE_ERR_NULL_PTR;
    }
    det->statistic = 0.0;
    return SEC_CHARGE_OK;
}

int sec_cusum_update(sec_cusum_t *det, double reported_feeder_kw, double expected_feeder_kw, sec_detector_alert_t *alert) {
    if (det == NULL || alert == NULL) {
        return SEC_CHARGE_ERR_NULL_PTR;
    }

    /* Under-reporting attack hides load: expected > reported => residual > 0 */
    double residual = expected_feeder_kw - reported_feeder_kw;

    /* S_k = max(0, S_{k-1} + r_k - k) */
    double candidate = det->statistic + residual - det->slack_kw;
    det->statistic = (candidate > 0.0) ? candidate : 0.0;

    bool is_alarm = (det->statistic > det->threshold_kw) || (residual > det->instant_threshold_kw);

    alert->is_alarm = is_alarm;
    alert->residual_kw = residual;
    alert->test_statistic = det->statistic;

    return SEC_CHARGE_OK;
}
