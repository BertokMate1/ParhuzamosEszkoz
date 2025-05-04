#include "../include/sha256.h"
#include <stdio.h>

char* load_kernel_source(const char *filename, size_t *source_size) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to open kernel file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    *source_size = ftell(file);
    rewind(file);

    char *source = (char*)malloc(*source_size + 1);
    fread(source, 1, *source_size, file);
    source[*source_size] = '\0';
    fclose(file);

    return source;
}

void sha256_hash(const char *input, uint32_t *digest) {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem input_buffer, output_buffer;
    cl_int err;

    // Load kernel source
    size_t source_size;
    char *kernel_source = load_kernel_source("kernel/sha256_kernel.cl", &source_size);

    // Get platform
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_CL_ERROR(err);

    // Get device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err == CL_DEVICE_NOT_FOUND) {
        printf("No GPU found, trying CPU...\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    }
    CHECK_CL_ERROR(err);

    // Create context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL_ERROR(err);

    // Create command queue
    queue = clCreateCommandQueue(context, device, NULL, &err);
    CHECK_CL_ERROR(err);

    // Create program
    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &source_size, &err);
    CHECK_CL_ERROR(err);

    // Build program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build failed:\n%s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }

    // Create kernel
    kernel = clCreateKernel(program, "sha256_hash", &err);
    CHECK_CL_ERROR(err);

    // Prepare input
    size_t length = strlen(input);
    size_t padded_length = (length < 56) ? 64 : 128;
    unsigned char *padded_input = (unsigned char*)calloc(padded_length, 1);
    
    if (length > 0) {
        memcpy(padded_input, input, length);
    }
    padded_input[length] = 0x80;

    uint64_t bit_length = length * 8;
    for (int i = 0; i < 8; i++) {
        padded_input[padded_length - 8 + i] = (bit_length >> (56 - i * 8)) & 0xFF;
    }

    // Create buffers
    input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 padded_length, padded_input, &err);
    CHECK_CL_ERROR(err);

    output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 8 * sizeof(uint32_t), NULL, &err);
    CHECK_CL_ERROR(err);

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    CHECK_CL_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
    CHECK_CL_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(uint), &length);
    CHECK_CL_ERROR(err);

    // Execute kernel
    size_t global_size = 1;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    // Read results
    err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, 8 * sizeof(uint32_t), digest, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    // Cleanup
    free(padded_input);
    free(kernel_source);
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

void hash_to_hex(const uint32_t *hash, char *output) {
    for (int i = 0; i < 8; i++) {
        sprintf(output + i * 8, "%08x", hash[i]);
    }
    output[64] = '\0';
}

void clean_input(char *str) {
    char *p = str;
    while (*p) {
        if (*p == '\n' || *p == '\r') *p = '\0';
        p++;
    }
}