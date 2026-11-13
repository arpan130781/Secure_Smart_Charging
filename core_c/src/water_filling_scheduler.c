/**
 * @file water_filling_scheduler.c
 * @brief Implementation of deterministic priority water-filling scheduler in ANSI C99.
 */

#include "water_filling_scheduler.h"
#include <math.h>

double sec_schedule_water_filling(
    const sec_feeder_state_t *feeder,
    sec_ev_session_t *sessions,
    size_t num_sessions,
    bool fail_safe_active,
    double safe_cap_kw
) {
    if (feeder == NULL || sessions == NULL) {
        return (double)SEC_CHARGE_ERR_NULL_PTR;
    }
    if (num_sessions > SEC_CHARGE_MAX_SESSIONS) {
        return (double)SEC_CHARGE_ERR_BOUNDS;
    }
    if (feeder->time_step_hours <= 0.0 || !isfinite(feeder->time_step_hours)) {
        return (double)SEC_CHARGE_ERR_PARAM;
    }

    /* Initialize all outputs to zero */
    for (size_t i = 0; i < num_sessions; ++i) {
        sessions[i].allocated_power_kw = 0.0;
    }

    if (num_sessions == 0) {
        return 0.0;
    }

    /* 1. Calculate physical transformer headroom */
    double dt = feeder->time_step_hours;
    double raw_headroom = feeder->transformer_limit_kw - feeder->base_load_kw - feeder->safety_margin_kw;
    double available_headroom = (raw_headroom > 0.0) ? raw_headroom : 0.0;

    /* Enforce cyber fail-safe safe cap if active */
    if (fail_safe_active) {
        double cap = (safe_cap_kw > 0.0) ? safe_cap_kw : 0.0;
        if (available_headroom > cap) {
            available_headroom = cap;
        }
    }

    if (available_headroom <= 0.0) {
        /* Zero headroom available: all EVs set to 0 kW */
        return 0.0;
    }

    /* 2. Compute individual vehicle power ceilings (rate, energy demand, dwell) */
    double p_cap[SEC_CHARGE_MAX_SESSIONS];
    double weights[SEC_CHARGE_MAX_SESSIONS];
    bool is_frozen[SEC_CHARGE_MAX_SESSIONS];
    double total_cap_demand = 0.0;

    for (size_t i = 0; i < num_sessions; ++i) {
        is_frozen[i] = false;
        double rem_e = sessions[i].remaining_energy_kwh;
        if (rem_e < 0.0 || !isfinite(rem_e)) {
            rem_e = 0.0;
        }

        double max_p = sessions[i].max_rate_kw;
        if (max_p < 0.0 || !isfinite(max_p)) {
            max_p = 0.0;
        }

        double dwell = sessions[i].dwell_hours;
        if (dwell <= 0.0 || !isfinite(dwell) || rem_e <= 1e-6) {
            p_cap[i] = 0.0;
            weights[i] = 0.0;
            is_frozen[i] = true;
            continue;
        }

        /* Power cannot exceed hardware rate nor remaining energy across time step */
        double energy_cap_kw = rem_e / dt;
        p_cap[i] = (max_p < energy_cap_kw) ? max_p : energy_cap_kw;

        /* Urgency weight: average power needed to satisfy session before departure */
        double eff_dwell = (dwell > dt) ? dwell : dt;
        double w = rem_e / eff_dwell;
        weights[i] = (w > 0.001) ? w : 0.001;

        total_cap_demand += p_cap[i];
    }

    /* If total unconstrained demand fits within available headroom, give all full power */
    if (total_cap_demand <= available_headroom) {
        double sum_allocated = 0.0;
        for (size_t i = 0; i < num_sessions; ++i) {
            sessions[i].allocated_power_kw = p_cap[i];
            sum_allocated += p_cap[i];
        }
        return sum_allocated;
    }

    /* 3. Priority Water-Filling Iteration */
    double rem_headroom = available_headroom;

    for (size_t iter = 0; iter < num_sessions; ++iter) {
        double sum_active_weights = 0.0;
        for (size_t i = 0; i < num_sessions; ++i) {
            if (!is_frozen[i]) {
                sum_active_weights += weights[i];
            }
        }

        if (sum_active_weights <= 0.0 || rem_headroom <= 0.0) {
            break;
        }

        bool clipped_any = false;
        for (size_t i = 0; i < num_sessions; ++i) {
            if (is_frozen[i]) {
                continue;
            }

            /* Proportional trial allocation based on urgency */
            double trial_p = (weights[i] / sum_active_weights) * rem_headroom;
            if (trial_p >= p_cap[i]) {
                /* Exceeds vehicle ceiling: freeze at cap and remove from remaining pool */
                sessions[i].allocated_power_kw = p_cap[i];
                rem_headroom -= p_cap[i];
                is_frozen[i] = true;
                clipped_any = true;
            }
        }

        if (!clipped_any) {
            /* All active sessions are strictly within their caps: distribute remainder */
            for (size_t i = 0; i < num_sessions; ++i) {
                if (!is_frozen[i]) {
                    sessions[i].allocated_power_kw = (weights[i] / sum_active_weights) * rem_headroom;
                    is_frozen[i] = true;
                }
            }
            break;
        }
    }

    /* Sum and return total dispatched power */
    double final_total = 0.0;
    for (size_t i = 0; i < num_sessions; ++i) {
        final_total += sessions[i].allocated_power_kw;
    }

    return final_total;
}
