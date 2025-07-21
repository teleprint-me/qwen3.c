/**
 * @file src/xorshift.c
 */

#include "xorshift.h"

uint32_t xorshift_int32(uint64_t* state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float xorshift_float(uint64_t* state) {
    return (xorshift_int32(state) >> 8) / 16777216.0f;
}
