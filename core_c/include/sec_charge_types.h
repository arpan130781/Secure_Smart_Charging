/**
 * @file sec_charge_types.h
 * @brief Common types and definitions for Embedded Secure Smart EV Charging.
 *
 * Designed for MISRA-C / ANSI C99 compliance with zero dynamic heap allocation.
 */

#ifndef SEC_CHARGE_TYPES_H
#define SEC_CHARGE_TYPES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SEC_CHARGE_MAX_SESSIONS 64U
#define SEC_CHARGE_OK           0
#define SEC_CHARGE_ERR_NULL_PTR -1
#define SEC_CHARGE_ERR_BOUNDS   -2
#define SEC_CHARGE_ERR_PARAM    -3

/**
 * @brief Status indicator for cyber detector alerts.
 */
typedef struct {
    bool is_alarm;          /**< True if anomaly threshold exceeded */
    double residual_kw;     /**< Innovation residual in kilowatts [kW] */
    double test_statistic;  /**< Value of CUSUM statistic or Chi-Square distance */
} sec_detector_alert_t;

/**
 * @brief EV Charging Session descriptor for embedded scheduler.
 */
typedef struct {
    uint32_t session_id;        /**< Numeric identifier */
    double max_rate_kw;         /**< Maximum hardware charging power [kW] */
    double energy_requested_kwh;/**< Total energy requested [kWh] */
    double energy_delivered_kwh;/**< Energy delivered so far [kWh] */
    double remaining_energy_kwh;/**< Unmet energy demand [kWh] */
    double dwell_hours;         /**< Time remaining until departure [h] */
    double allocated_power_kw;  /**< Output: commanded charging power [kW] */
} sec_ev_session_t;

/**
 * @brief Feeder grid constraints and base load snapshot.
 */
typedef struct {
    double transformer_limit_kw; /**< Continuous thermal limit of feeder [kW] */
    double base_load_kw;         /**< Non-EV background load [kW] */
    double safety_margin_kw;     /**< Reserved headroom buffer [kW] */
    double time_step_hours;      /**< Control dispatch interval [h] (e.g. 0.25 for 15 min) */
} sec_feeder_state_t;

#ifdef __cplusplus
}
#endif

#endif /* SEC_CHARGE_TYPES_H */
