/**
 * @file cusum.h
 * @brief Classical Cumulative Sum (CUSUM) sequential change-point detector.
 *
 * Implements Page's CUSUM algorithm for detecting under-reporting False Data
 * Injection (FDI) cyber-attacks on power grid feeder meters.
 */

#ifndef SEC_CUSUM_H
#define SEC_CUSUM_H

#include "sec_charge_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double threshold_kw;        /**< Cumulative decision threshold h [kW] */
    double slack_kw;            /**< Drift compensation parameter k [kW] */
    double instant_threshold_kw;/**< Single-step large anomaly cutoff [kW] */
    double statistic;           /**< Cumulative test statistic S_k [kW] */
} sec_cusum_t;

/**
 * @brief Initialize CUSUM detector state.
 */
int sec_cusum_init(sec_cusum_t *det, double threshold_kw, double slack_kw, double instant_threshold_kw);

/**
 * @brief Reset cumulative statistic to zero.
 */
int sec_cusum_reset(sec_cusum_t *det);

/**
 * @brief Ingest sample and evaluate sequential decision rule.
 *
 * @param det Pointer to detector state
 * @param reported_feeder_kw Power telemetry read from meter [kW]
 * @param expected_feeder_kw Physics-based expected feeder load [kW]
 * @param alert Output alert structure
 * @return 0 on success, negative error code on error
 */
int sec_cusum_update(sec_cusum_t *det, double reported_feeder_kw, double expected_feeder_kw, sec_detector_alert_t *alert);

#ifdef __cplusplus
}
#endif

#endif /* SEC_CUSUM_H */
