/**
 * @file secoc.c
 * @brief Standalone implementation of AUTOSAR SecOC CAN FD protocol in ANSI C99.
 *
 * Includes self-contained SHA-256 and HMAC-SHA256 with zero external dependencies.
 */

#include "secoc.h"
#include <string.h>

/* --- Embedded SHA-256 / HMAC-SHA256 Engine --- */

typedef struct {
    uint32_t state[8];
    uint64_t count;
    uint8_t buffer[64];
} sec_sha256_ctx_t;

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static void sha256_transform(sec_sha256_ctx_t *ctx, const uint8_t data[64]) {
    uint32_t a, b, c, d, e, f, g, h, t1, t2, m[64];
    for (size_t i = 0, j = 0; i < 16; ++i, j += 4) {
        m[i] = ((uint32_t)data[j] << 24) | ((uint32_t)data[j + 1] << 16) |
               ((uint32_t)data[j + 2] << 8) | ((uint32_t)data[j + 3]);
    }
    for (size_t i = 16; i < 64; ++i) {
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];
    }
    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2]; d = ctx->state[3];
    e = ctx->state[4]; f = ctx->state[5]; g = ctx->state[6]; h = ctx->state[7];

    for (size_t i = 0; i < 64; ++i) {
        t1 = h + EP1(e) + CH(e, f, g) + K[i] + m[i];
        t2 = EP0(a) + MAJ(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

static void sha256_init(sec_sha256_ctx_t *ctx) {
    ctx->count = 0;
    ctx->state[0] = 0x6a09e667; ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372; ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f; ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab; ctx->state[7] = 0x5be0cd19;
}

static void sha256_update(sec_sha256_ctx_t *ctx, const uint8_t *data, size_t len) {
    size_t buffer_idx = (size_t)(ctx->count & 0x3F);
    ctx->count += len;
    for (size_t i = 0; i < len; ++i) {
        ctx->buffer[buffer_idx++] = data[i];
        if (buffer_idx == 64) {
            sha256_transform(ctx, ctx->buffer);
            buffer_idx = 0;
        }
    }
}

static void sha256_final(sec_sha256_ctx_t *ctx, uint8_t hash[32]) {
    uint8_t pad = 0x80;
    size_t buffer_idx = (size_t)(ctx->count & 0x3F);
    uint64_t bit_len = ctx->count * 8;

    sha256_update(ctx, &pad, 1);
    buffer_idx = (size_t)(ctx->count & 0x3F);
    if (buffer_idx > 56) {
        while (buffer_idx < 64) ctx->buffer[buffer_idx++] = 0;
        sha256_transform(ctx, ctx->buffer);
        buffer_idx = 0;
    }
    while (buffer_idx < 56) ctx->buffer[buffer_idx++] = 0;
    for (int i = 7; i >= 0; --i) {
        ctx->buffer[56 + (7 - i)] = (uint8_t)(bit_len >> (i * 8));
    }
    sha256_transform(ctx, ctx->buffer);

    for (size_t i = 0; i < 8; ++i) {
        hash[i * 4]     = (uint8_t)(ctx->state[i] >> 24);
        hash[i * 4 + 1] = (uint8_t)(ctx->state[i] >> 16);
        hash[i * 4 + 2] = (uint8_t)(ctx->state[i] >> 8);
        hash[i * 4 + 3] = (uint8_t)(ctx->state[i]);
    }
}

static void hmac_sha256(
    const uint8_t *key, size_t key_len,
    const uint8_t *msg, size_t msg_len,
    uint8_t out[32]
) {
    uint8_t k_pad[64];
    uint8_t k[64];
    memset(k, 0, 64);
    if (key_len > 64) {
        sec_sha256_ctx_t ctx;
        sha256_init(&ctx);
        sha256_update(&ctx, key, key_len);
        sha256_final(&ctx, k);
    } else {
        memcpy(k, key, key_len);
    }

    sec_sha256_ctx_t ctx;

    /* Inner hash: H((K ^ ipad) || msg) */
    for (size_t i = 0; i < 64; ++i) k_pad[i] = k[i] ^ 0x36;
    sha256_init(&ctx);
    sha256_update(&ctx, k_pad, 64);
    sha256_update(&ctx, msg, msg_len);
    uint8_t inner_hash[32];
    sha256_final(&ctx, inner_hash);

    /* Outer hash: H((K ^ opad) || inner_hash) */
    for (size_t i = 0; i < 64; ++i) k_pad[i] = k[i] ^ 0x5c;
    sha256_init(&ctx);
    sha256_update(&ctx, k_pad, 64);
    sha256_update(&ctx, inner_hash, 32);
    sha256_final(&ctx, out);
}

/* --- Serialization Helpers --- */

static void put_u32_be(uint8_t *buf, uint32_t val) {
    buf[0] = (uint8_t)(val >> 24);
    buf[1] = (uint8_t)(val >> 16);
    buf[2] = (uint8_t)(val >> 8);
    buf[3] = (uint8_t)(val);
}

static uint32_t get_u32_be(const uint8_t *buf) {
    return ((uint32_t)buf[0] << 24) | ((uint32_t)buf[1] << 16) |
           ((uint32_t)buf[2] << 8)  | ((uint32_t)buf[3]);
}

static void put_float_be(uint8_t *buf, float val) {
    uint32_t u;
    memcpy(&u, &val, sizeof(float));
    put_u32_be(buf, u);
}

static float get_float_be(const uint8_t *buf) {
    uint32_t u = get_u32_be(buf);
    float val;
    memcpy(&val, &u, sizeof(float));
    return val;
}

/* Constant-time memory comparison to mitigate timing side-channels */
static bool constant_time_eq(const uint8_t *a, const uint8_t *b, size_t len) {
    uint8_t diff = 0;
    for (size_t i = 0; i < len; ++i) {
        diff |= (a[i] ^ b[i]);
    }
    return (diff == 0);
}

/* --- Public SecOC API --- */

int secoc_verifier_init(secoc_verifier_t *secoc, const uint8_t key[SECOC_KEY_LEN_BYTES], uint32_t epoch) {
    if (secoc == NULL || key == NULL) {
        return SEC_CHARGE_ERR_NULL_PTR;
    }
    memcpy(secoc->key, key, SECOC_KEY_LEN_BYTES);
    secoc->current_epoch = epoch;
    secoc->last_counter = 0;
    secoc->initialized = true;
    return SEC_CHARGE_OK;
}

int secoc_build_frame(
    const uint8_t key[SECOC_KEY_LEN_BYTES],
    uint32_t epoch,
    uint32_t counter,
    const secoc_bms_payload_t *bms,
    uint8_t *out_frame,
    size_t *out_frame_len
) {
    if (key == NULL || bms == NULL || out_frame == NULL || out_frame_len == NULL) {
        return SEC_CHARGE_ERR_NULL_PTR;
    }

    /* Total payload: 4(epoch) + 4(counter) + 16(floats) + 1(flags) = 25 bytes.
     * With 8 bytes MAC = 33 bytes, or pad to 24 payload + 8 MAC = 32 bytes (CAN FD standard).
     * Let's use:
     * 0..3:   epoch
     * 4..7:   counter
     * 8..11:  voltage
     * 12..15: current
     * 16..19: soc
     * 20..23: temp
     * 24:     fault_flags
     * Total authenticated data = 25 bytes. MAC = 8 bytes -> 33 bytes.
     */
    put_u32_be(out_frame + 0, epoch);
    put_u32_be(out_frame + 4, counter);
    put_float_be(out_frame + 8, bms->battery_voltage_v);
    put_float_be(out_frame + 12, bms->battery_current_a);
    put_float_be(out_frame + 16, bms->state_of_charge_pct);
    put_float_be(out_frame + 20, bms->temperature_c);
    out_frame[24] = bms->fault_flags;

    /* Compute HMAC-SHA256 over the 25 authenticated bytes */
    uint8_t full_mac[32];
    hmac_sha256(key, SECOC_KEY_LEN_BYTES, out_frame, 25, full_mac);

    /* Append truncated 8-byte (64-bit) SecOC MAC */
    memcpy(out_frame + 25, full_mac, SECOC_MAC_LEN_BYTES);
    *out_frame_len = 25 + SECOC_MAC_LEN_BYTES; /* 33 bytes */

    return SEC_CHARGE_OK;
}

int secoc_verify_and_unpack(
    secoc_verifier_t *secoc,
    const uint8_t *frame,
    size_t frame_len,
    secoc_bms_payload_t *out_bms
) {
    if (secoc == NULL || frame == NULL || out_bms == NULL) {
        return SEC_CHARGE_ERR_NULL_PTR;
    }
    if (!secoc->initialized) {
        return SEC_CHARGE_ERR_PARAM;
    }
    if (frame_len < (25 + SECOC_MAC_LEN_BYTES)) {
        return SEC_CHARGE_ERR_BOUNDS;
    }

    /* 1. Extract and check freshness: epoch and monotonic counter */
    uint32_t rx_epoch = get_u32_be(frame + 0);
    uint32_t rx_counter = get_u32_be(frame + 4);

    if (rx_epoch < secoc->current_epoch) {
        /* Rollback attack: older epoch rejected */
        return -10;
    }

    if (rx_epoch == secoc->current_epoch && rx_counter <= secoc->last_counter) {
        /* Replay attack: non-increasing counter rejected */
        return -11;
    }

    /* 2. Cryptographic MAC verification */
    uint8_t full_mac[32];
    hmac_sha256(secoc->key, SECOC_KEY_LEN_BYTES, frame, 25, full_mac);

    if (!constant_time_eq(frame + 25, full_mac, SECOC_MAC_LEN_BYTES)) {
        /* Forged payload or corrupted MAC */
        return -12;
    }

    /* 3. Acceptance: commit monotonic state and unpack payload */
    secoc->current_epoch = rx_epoch;
    secoc->last_counter = rx_counter;

    out_bms->battery_voltage_v = get_float_be(frame + 8);
    out_bms->battery_current_a = get_float_be(frame + 12);
    out_bms->state_of_charge_pct = get_float_be(frame + 16);
    out_bms->temperature_c = get_float_be(frame + 20);
    out_bms->fault_flags = frame[24];

    return SEC_CHARGE_OK;
}
