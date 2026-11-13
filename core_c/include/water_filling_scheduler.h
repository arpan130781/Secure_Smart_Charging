/**
 * @file water_filling_scheduler.h
 * @brief Deterministic O(N) Priority Water-Filling Charging Scheduler.
 *
 * Designed for microcontrollers with <32 KB RAM and zero dynamic heap allocation.
 * Computes optimal/sub-optimal EV charging power setpoints bounded by physical
 * transformer headroom, vehicle maximum rates, and remaining energy requests.
 */

#ifndef SEC_WATER_FILLING_SCHEDULER_H
#define SEC_WATER_FILLING_SCHEDULER_H

#include "sec_charge_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Dispatch charging setpoints using deterministic priority water-filling.
 *
 * Enforces:
 *  1. Non-negativity: 0 <= P_i <= max_rate_kw
 *  2. Energy conservation: P_i * time_step <= remaining_energy_kwh
 *  3. Feeder thermal limit: sum(P_i) <= max(0, transformer_limit - base_load - margin)
 *  4. Latch safe-state: If fail_safe_active is true, restricts total EV power to safe_cap_kw.
 *
 * @param feeder Feeder capacity, current base load, and time step
 * @param sessions Array of EV sessions (modified in-place with allocated_power_kw)
 * @param num_sessions Number of sessions in array (must be <= SEC_CHARGE_MAX_SESSIONS)
 * @param fail_safe_active True if cyber-detector raised alarm or meter faulted
 * @param safe_cap_kw Conservative fallback aggregate cap when in fail-safe (e.g. 0.0 or fixed limit)
 * @return Total power allocated to all EVs in kW (>= 0.0), or negative on error
 */
double sec_schedule_water_filling(
    const sec_feeder_state_t *feeder,
    sec_ev_session_t *sessions,
    size_t num_sessions,
    bool fail_safe_active,
    double safe_cap_kw
);

#ifdef __cplusplus
}
#endif

#endif /* SEC_WATER_FILLING_SCHEDULER_H */
