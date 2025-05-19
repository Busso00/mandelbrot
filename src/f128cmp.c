#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>

typedef struct {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    uint64_t lo;  // least significant 64 bits (lower address)
    uint64_t hi;  // most significant 64 bits (higher address)
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    uint64_t hi;  // most significant 64 bits (lower address)
    uint64_t lo;  // least significant 64 bits (higher address)
#else
#error "Unknown endianness"
#endif
} f128_bits_t;

f128_bits_t float128_to_bits(__float128 f) {
    f128_bits_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return bits;
}

void float128_to_ordered(f128_bits_t* bits) {
    // Extract sign bit (bit 127)
    uint64_t sign = bits->hi >> 63;

    if (sign) {
        // Negative numbers: flip all bits for two's complement ordering
        bits->hi = ~bits->hi;
        bits->lo = ~bits->lo;
    } else {
        // Positive numbers: set the sign bit to maintain order
        bits->hi |= 0x8000000000000000ULL;
    }
}

bool f128_eq(__float128 a, __float128 b) {
    f128_bits_t ba = float128_to_bits(a);
    f128_bits_t bb = float128_to_bits(b);
    return ba.hi == bb.hi && ba.lo == bb.lo;
}

bool f128_lt(__float128 a, __float128 b) {
    f128_bits_t ba = float128_to_bits(a);
    f128_bits_t bb = float128_to_bits(b);
    float128_to_ordered(&ba);
    float128_to_ordered(&bb);

    if (ba.hi < bb.hi) return true;
    if (ba.hi > bb.hi) return false;
    return ba.lo < bb.lo;
}

bool f128_gt(__float128 a, __float128 b) {
    f128_bits_t ba = float128_to_bits(a);
    f128_bits_t bb = float128_to_bits(b);
    float128_to_ordered(&ba);
    float128_to_ordered(&bb);

    if (ba.hi > bb.hi) return true;
    if (ba.hi < bb.hi) return false;
    return ba.lo > bb.lo;
}

int f128cmp(__float128 a, __float128 b) {
    assert(
        (f128_eq(a, b) && !f128_lt(a, b) && !f128_gt(a, b)) ||
        (f128_lt(a, b) && !f128_eq(a, b) && !f128_gt(a, b)) ||
        (f128_gt(a, b) && !f128_eq(a, b) && !f128_lt(a, b))
    );

    if (f128_lt(a, b)) return -1;
    if (f128_eq(a, b)) return 0;
    return 1;
}
