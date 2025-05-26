#ifndef SHA256_H
#define SHA256_H
#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define CHECK_CL_ERROR(err) \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL Error %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

typedef unsigned int uint;

void sha256_hash(const char *input, uint32_t *digest);
void hash_to_hex(const uint32_t *hash, char *output);
void clean_input(char *str);

#endif
