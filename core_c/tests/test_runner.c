/**
 * @file test_runner.c
 * @brief Standalone ANSI C unit test suite for Embedded Core modules.
 */

#include "cusum.h"
#include "kalman_chisq.h"
#include "water_filling_scheduler.h"
#include "secoc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define TEST_ASSERT(cond, msg) do { \
    g_tests_run++; \
    if (!(cond)) { \
        printf("FAIL: %s (line %d): %s\n", __func__, __LINE__, msg); \
    } else { \
        g_tests_passed++; \
    } \
} while (0)

/* --- CUSUM Tests --- */
static void test_cusum_normal_and_attack(void) {
    sec_cusum_t det;
    sec_detector_alert_t alert;
    int rc = sec_cusum_init(&det, 18.0, 1.5, 20.0);
    TEST_ASSERT(rc == SEC_CHARGE_OK, "CUSUM init success");

    /* 1. Normal operation: reported == expected => residual == 0 */
    for (int i = 0; i < 20; ++i) {
        rc = sec_cusum_update(&det, 50.0, 50.0, &alert);
        TEST_ASSERT(rc == SEC_CHARGE_OK, "CUSUM update success");
        TEST_ASSERT(!alert.is_alarm, "Normal operation raises no alarm");
        TEST_ASSERT(alert.test_statistic == 0.0, "Statistic stays at 0");
    }

    /* 2. Attack injection: under-reporting by 35 kW */
    bool alarmed = false;
    for (int i = 0; i < 5; ++i) {
        sec_cusum_update(&det, 15.0, 50.0, &alert);
        if (alert.is_alarm) {
            alarmed = true;
            break;
        }
    }
    TEST_ASSERT(alarmed, "CUSUM quickly triggers alarm on under-reporting attack");
}

/* --- Kalman Filter & Chi-Square Tests --- */
static void test_kalman_chisq_normal_and_attack(void) {
    sec_kalman_chisq_t kf;
    sec_detector_alert_t alert;
    int rc = sec_kalman_chisq_init(&kf, 0.5, 0.09, 9.0, 0.3);
    TEST_ASSERT(rc == SEC_CHARGE_OK, "Kalman Chi-Sq init success");

    /* Normal small noise around 50 kW */
    for (int i = 0; i < 30; ++i) {
        double noise = ((i % 3) - 1) * 0.1;
        sec_kalman_chisq_update(&kf, 50.0 + noise, 50.0, &alert);
        TEST_ASSERT(!alert.is_alarm, "Normal noise does not trigger Chi-Sq alarm");
    }

    /* Attack: sudden 25 kW jump/under-reporting */
    sec_kalman_chisq_update(&kf, 25.0, 50.0, &alert);
    TEST_ASSERT(alert.is_alarm, "Sudden 25 kW anomaly triggers Chi-Sq alarm");
    TEST_ASSERT(alert.test_statistic > 9.0, "Chi-Square distance exceeds threshold");
}

/* --- Water-Filling Scheduler Tests --- */
static void test_water_filling_scheduler(void) {
    sec_feeder_state_t feeder = {
        .transformer_limit_kw = 100.0,
        .base_load_kw = 40.0,
        .safety_margin_kw = 5.0,
        .time_step_hours = 0.25 /* 15 min */
    };
    /* Available headroom = 100 - 40 - 5 = 55 kW */

    sec_ev_session_t sessions[3] = {
        {
            .session_id = 1,
            .max_rate_kw = 22.0,
            .remaining_energy_kwh = 20.0,
            .dwell_hours = 2.0
        },
        {
            .session_id = 2,
            .max_rate_kw = 22.0,
            .remaining_energy_kwh = 30.0,
            .dwell_hours = 1.0 /* More urgent */
        },
        {
            .session_id = 3,
            .max_rate_kw = 22.0,
            .remaining_energy_kwh = 10.0,
            .dwell_hours = 4.0
        }
    };

    /* 1. Normal run under headroom constraint */
    double total = sec_schedule_water_filling(&feeder, sessions, 3, false, 0.0);
    TEST_ASSERT(total <= 55.0001, "Allocated total never exceeds headroom (55 kW)");
    TEST_ASSERT(total > 45.0, "Allocated total efficiently utilizes capacity");

    /* Vehicle 2 is most urgent: should get its maximum rate */
    TEST_ASSERT(sessions[1].allocated_power_kw >= sessions[0].allocated_power_kw, "Urgent EV gets higher priority");

    /* 2. Fail-safe active with safe cap of 10.0 kW */
    double safe_total = sec_schedule_water_filling(&feeder, sessions, 3, true, 10.0);
    TEST_ASSERT(safe_total <= 10.0001, "Fail-safe clamps power strictly to safe cap");
}

/* --- SecOC Tests --- */
static void test_secoc_protocol(void) {
    uint8_t key[SECOC_KEY_LEN_BYTES];
    memset(key, 0x42, SECOC_KEY_LEN_BYTES);

    secoc_verifier_t verifier;
    int rc = secoc_verifier_init(&verifier, key, 1);
    TEST_ASSERT(rc == SEC_CHARGE_OK, "SecOC verifier init success");

    secoc_bms_payload_t tx_bms = {
        .battery_voltage_v = 380.5f,
        .battery_current_a = 45.0f,
        .state_of_charge_pct = 65.2f,
        .temperature_c = 28.4f,
        .fault_flags = 0x00
    };

    uint8_t frame[64];
    size_t frame_len = 0;
    rc = secoc_build_frame(key, 1, 100, &tx_bms, frame, &frame_len);
    TEST_ASSERT(rc == SEC_CHARGE_OK, "Frame built successfully");
    TEST_ASSERT(frame_len == 33, "SecOC frame length is exactly 33 bytes");

    /* 1. Genuine frame verification */
    secoc_bms_payload_t rx_bms;
    rc = secoc_verify_and_unpack(&verifier, frame, frame_len, &rx_bms);
    TEST_ASSERT(rc == SEC_CHARGE_OK, "Genuine frame accepted");
    TEST_ASSERT(fabsf(rx_bms.battery_voltage_v - tx_bms.battery_voltage_v) < 1e-4f, "Voltage matches");

    /* 2. Replay attack rejection */
    rc = secoc_verify_and_unpack(&verifier, frame, frame_len, &rx_bms);
    TEST_ASSERT(rc == -11, "Replayed counter strictly rejected");

    /* 3. Tampering / Forgery rejection */
    frame[4] = 101; /* newer counter */
    frame[8] ^= 0xFF; /* tamper voltage payload */
    rc = secoc_verify_and_unpack(&verifier, frame, frame_len, &rx_bms);
    TEST_ASSERT(rc == -12, "Forged MAC strictly rejected");

    /* 4. Rollback epoch rejection */
    rc = secoc_build_frame(key, 0, 500, &tx_bms, frame, &frame_len); /* older epoch 0 */
    rc = secoc_verify_and_unpack(&verifier, frame, frame_len, &rx_bms);
    TEST_ASSERT(rc == -10, "Epoch rollback strictly rejected");
}

int main(void) {
    printf("====================================================================\n");
    printf("RUNNING EMBEDDED C99 CORE UNIT TESTS (ZERO-HEAP, ANSI C)\n");
    printf("====================================================================\n");

    test_cusum_normal_and_attack();
    test_kalman_chisq_normal_and_attack();
    test_water_filling_scheduler();
    test_secoc_protocol();

    printf("--------------------------------------------------------------------\n");
    printf("C Test Results: %d/%d assertions passed.\n", g_tests_passed, g_tests_run);
    printf("====================================================================\n");

    return (g_tests_run == g_tests_passed) ? 0 : 1;
}
