#include "../include/sha256.h"
#include <stdio.h>
#include <ctype.h>

int main() {
    char input[1024];
    char hex[65];
    uint32_t digest[8];

    printf("=== SHA-256 Hasher ===\n");
    printf("Type 'exit' to quit\n\n");

  while (1) {
        printf("Enter text to hash: ");
        fflush(stdout);
        
        if (!fgets(input, sizeof(input), stdin)) break;
        clean_input(input);
		
        if (strcmp(input, "exit") == 0) break;

        sha256_hash(input, digest);
        hash_to_hex(digest, hex);
        
        printf("SHA-256: %s\n\n", hex);
    }

    printf("Goodbye!\n");
    return 0;
}