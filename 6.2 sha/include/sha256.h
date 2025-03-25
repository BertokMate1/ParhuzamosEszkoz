#ifndef SHA256_H
#define SHA256_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <CL/cl.h>

void verify_initial_hash_values();
void verify_round_constants();

void sha256_hash(const char *input, uint32_t *digest);
void hash_to_hex(const uint32_t *hash, char *output);

void run_unit_tests();

#endif