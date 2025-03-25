#include "../include/sha256.h"

int main() {
    verify_initial_hash_values();
    verify_round_constants();
    run_unit_tests();
    
    const char *test_str = "hello world";
    uint32_t digest[8];
    char hex[65];
    
    sha256_hash(test_str, digest);
    hash_to_hex(digest, hex);
    
    printf("\nExample Hash: %s\n", hex);
    return 0;
}