#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <math.h>

#define MAX_SOURCE_SIZE (0x100000)

void matrixMultip(const float* A, const float* B, float* C, const int M, const int N, const int K) {
	for (int m=0; m<M; ++m) {
		for (int n=0; n<N; ++n) {
			float acc = 0.0f;
			
			for (int k=0; k<K; ++k) {
				acc += A[m * M + k] * B[k * N + n];
			}
			C[m * M + n] = acc;
		}
	}
}

int main() {
	
	srand(2006);
	
	// Load the kernel from source file
	FILE *kernelFile;
	char *kernelSource;
	size_t kernelSize;
	
    kernelFile = fopen("matrixMultip.cl", "r");
    if (!kernelFile) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
	
	kernelSource = (char*)malloc(MAX_SOURCE_SIZE);
    kernelSize = fread(kernelSource, 1, MAX_SOURCE_SIZE, kernelFile);
    fclose(kernelFile);
	
	cl_platform_id platformId = NULL;
    cl_device_id deviceID = NULL;
	cl_command_queue commandQueue;
	cl_program program;
	cl_kernel kernel;
	cl_int err;
	cl_context context;
	
	// Get platform and device
    err = clGetPlatformIDs(1, &platformId, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to get platform ID.\n");
        return EXIT_FAILURE;
    }
    err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &deviceId, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to get device ID.\n");
        return EXIT_FAILURE;
    }
	
	    // Create context
    context = clCreateContext(NULL, 1, &deviceId, NULL, NULL, &err);
    if (!context || err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create context.\n");
        return EXIT_FAILURE;
    }

    // Create command queue
    commandQueue = clCreateCommandQueue(context, deviceId, 0, &err);
    if (!commandQueue || err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create command queue.\n");
        return EXIT_FAILURE;  
    }

    // Create program object
    program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource,
                                        (const size_t *)&kernelSize, &err);
    if (!program || err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create program object.\n");
        return EXIT_FAILURE;
    }

    // Build program
    err = clBuildProgram(program, 1, &deviceId, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to build program.\n");
        return EXIT_FAILURE;
    }
	
	// Create kernel object
    kernel = clCreateKernel(program, "matrixMultip", &err);
    if (!kernel || err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create kernel object.\n");
        return EXIT_FAILURE;
    }
	
	const int M = 5;
	const int N = 5;
	const int K = 5;
	
	float *A = (float *)malloc(sizeof(float) * M * K);
    float *B = (float *)malloc(sizeof(float) * K * N);
    float *C = (float *)malloc(sizeof(float) * M * N);
    float *D = (float *)malloc(sizeof(float) * M * N);
	
	for (int i = 0; i < K * M; ++i) {
		A[i] = rand() / (float)RAND_MAX;
	}
	
	for (int i = 0; i < K * N; ++i) {
		B[i] = rand() / (float)RAND_MAX;
	}
	
	cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * M * K, NULL, &err);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * K * N, NULL, &err);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * M * N, NULL, &err);
	
    if (!bufferA || !bufferB || !bufferC || err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create buffer objects.\n");
        return EXIT_FAILURE;
    }
	
	matrixMultip(A, B, D, M, N, K);
	
	err = clEnqueueWriteBuffer(commandQueue, bufferA, CL_TRUE, 0, sizeof(float) * M * K, A, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commandQueue, bufferB, CL_TRUE, 0, sizeof(float) * K * N, B, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to write data to device.\n");
        return EXIT_FAILURE;
    }
	
    fflush(stdout);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);
    free(kernelSource);
    free(A);
    free(B);
    free(C);
    printf("Cleaned up!\n");
    fflush(stdout);

    return EXIT_SUCCESS
	
}