#include "../include/sha256.h"

char* load_kernel_source(const char *filename, size_t *source_size) {
    FILE *file = fopen(filename, "r");
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
    char *kernel_source = load_kernel_source("../kernel/sha256_kernel.cl", &source_size);
    
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);
    
    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, 
                                      &source_size, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    
    size_t length = strlen(input);
    size_t padded_length = (length < 56) ? 64 : 128;
    uint8_t *padded_input = calloc(padded_length, 1);
    memcpy(padded_input, input, length);
    padded_input[length] = 0x80;
    
    uint64_t bit_length = length * 8;
    for(int i=0; i<8; i++) {
        padded_input[padded_length-8+i] = (bit_length >> (56 - i*8)) & 0xFF;
    }
    
    input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 padded_length, padded_input, &err);
    output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                                  8*sizeof(uint32_t), NULL, &err);
    
    kernel = clCreateKernel(program, "sha256_hash", &err);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
    clSetKernelArg(kernel, 2, sizeof(uint), &length);
    
    size_t global_size = 1;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    
    clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, 
                      8*sizeof(uint32_t), digest, 0, NULL, NULL);
    
    free(padded_input);
    free(kernel_source);
    clReleaseMemObject(input_buffer);
}

void hash_to_hex(const uint32_t *hash, char *output) {
    for(int i=0; i<8; i++) {
        sprintf(output + i*8, "%08x", hash[i]);
    }
    output[64] = '\0';
}