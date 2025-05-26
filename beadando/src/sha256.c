#include "../include/sha256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

char* load_kernel_source(const char *filename, size_t *source_size) {
    FILE *file = fopen(filename, "rb");
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

    size_t source_size;
    char *kernel_source = load_kernel_source("kernel/sha256_kernel.cl", &source_size);

    err = clGetPlatformIDs(1, &platform, NULL); CHECK_CL_ERROR(err);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err == CL_DEVICE_NOT_FOUND) {
        printf("No GPU found, trying CPU...\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    }
    CHECK_CL_ERROR(err);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err); CHECK_CL_ERROR(err);
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err); CHECK_CL_ERROR(err);

    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &source_size, &err); CHECK_CL_ERROR(err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size; clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build failed:\n%s\n", log);
        free(log); exit(EXIT_FAILURE);
    }
    kernel = clCreateKernel(program, "sha256_hash", &err); CHECK_CL_ERROR(err);

    size_t length = strlen(input);
    size_t padded_length = ((length + 9 + 63) / 64) * 64; // Fixed padding calculation
    unsigned char *padded_input = (unsigned char*)calloc(padded_length, 1);
    if (length) memcpy(padded_input, input, length);
    padded_input[length] = 0x80;
    uint64_t bit_length = length * 8;
    for (int i = 0; i < 8; i++) padded_input[padded_length - 8 + i] = (bit_length >> (56 - i*8)) & 0xFF;

    uint num_blocks = padded_length / 64; // Correct block count

    input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, padded_length, padded_input, &err); CHECK_CL_ERROR(err);
    output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 8 * sizeof(uint32_t), NULL, &err); CHECK_CL_ERROR(err);

    CHECK_CL_ERROR(clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer));
    CHECK_CL_ERROR(clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer));
    CHECK_CL_ERROR(clSetKernelArg(kernel, 2, sizeof(uint), &num_blocks));

    size_t global_size = 1; // Single work item for single input
    cl_event kernel_event;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, &kernel_event);
    CHECK_CL_ERROR(err);

    cl_event read_event;
    err = clEnqueueReadBuffer(queue, output_buffer, CL_FALSE, 0, 8 * sizeof(uint32_t), digest, 0, NULL, &read_event);
    CHECK_CL_ERROR(err);

    clWaitForEvents(1, &kernel_event);
    clWaitForEvents(1, &read_event);

    cl_ulong k_start, k_end;
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(k_start), &k_start, NULL);
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(k_end), &k_end, NULL);
    printf("Kernel runtime: %.3f ms\n", (k_end - k_start) * 1e-6);

    cl_ulong r_start, r_end;
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(r_start), &r_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(r_end), &r_end, NULL);
    printf("ReadBuffer runtime: %.3f ms\n", (r_end - r_start) * 1e-6);

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
