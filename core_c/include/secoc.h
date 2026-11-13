/**
 * @file secoc.h
 * @brief AUTOSAR SecOC (Secure Onboard Communication) CAN FD Protocol Layer.
 *
 * Implements cryptographic authentication and freshness verification for
 * CAN FD BMS messages with zero external dependencies.
 */

#ifndef SEC_SECOC_H
#define SEC_SECOC_H

#include "sec_charge_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SECOC_MAC_LEN_BYTES     8U   /**< Truncated 64-bit MAC as per SecOC profile */
#define SECOC_MAX_PAYLOAD_BYTES 64U  /**< Maximum CAN FD payload length */
#define SECOC_KEY_LEN_BYTES     32U  /**< 256-bit symmetric pre-shared key */

typedef struct {
    uint8_t key[SECOC_KEY_LEN_BYTES]; /**< Pre-shared secret key */
    uint32_t current_epoch;           /**< Boot / session epoch */
    uint32_t last_counter;            /**< Monotonic sequence counter */
    bool initialized;
} secoc_verifier_t;

/**
 * @brief BMS Status telemetry payload structure.
 */
typedef struct {
    float battery_voltage_v;   /**< Battery terminal voltage [V] */
    float battery_current_a;   /**< Active charging current [A] */
    float state_of_charge_pct; /**< State of Charge [0.0 - 100.0 %] */
    float temperature_c;       /**< Pack temperature [deg C] */
    uint8_t fault_flags;       /**< Diagnostic error bits */
} secoc_bms_payload_t;

/**
 * @brief Initialize SecOC verifier with symmetric key and starting epoch.
 */
int secoc_verifier_init(secoc_verifier_t *secoc, const uint8_t key[SECOC_KEY_LEN_BYTES], uint32_t epoch);

/**
 * @brief Build and authenticate a secured CAN FD BMS frame.
 *
 * @param key Pre-shared key
 * @param epoch Current boot/session epoch
 * @param counter Monotonic counter
 * @param bms Telemetry struct to pack
 * @param out_frame Output buffer (must be at least 32 bytes)
 * @param out_frame_len Output frame length written
 * @return 0 on success, negative on error
 */
int secoc_build_frame(
    const uint8_t key[SECOC_KEY_LEN_BYTES],
    uint32_t epoch,
    uint32_t counter,
    const secoc_bms_payload_t *bms,
    uint8_t *out_frame,
    size_t *out_frame_len
);

/**
 * @brief Verify and unpack a secured CAN FD BMS frame.
 *
 * @param secoc Verifier state
 * @param frame Inbound CAN FD frame bytes
 * @param frame_len Length of frame in bytes
 * @param out_bms Unpacked BMS telemetry
 * @return 0 if valid and fresh, negative error if forged, replayed, or corrupt
 */
int secoc_verify_and_unpack(
    secoc_verifier_t *secoc,
    const uint8_t *frame,
    size_t frame_len,
    secoc_bms_payload_t *out_bms
);

#ifdef __cplusplus
}
#endif

#endif /* SEC_SECOC_H */
