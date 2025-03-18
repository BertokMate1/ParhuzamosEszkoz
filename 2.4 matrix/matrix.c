#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <math.h>
#include <sys/time.h>

#define MAX_SOURCE_SIZE (0x100000)

void matrixMultip(const float* A, const float* B, float* C, const int M, const int N, const int K) {
	for (int m=0; m<M; ++m) {
		for (int n=0; n<N; ++n) {
			float acc = 0.0f;
			
			for (int k=0; k<K; ++k) {
				acc += A[m * K + k] * B[k * N + n];
			}
			C[m * M + n] = acc;
		}
	}
}

double getTimeStamp() {
	struct timeval tv;
    gettimeofday(&tv, NULL);
	return ((double)tv.tv_sec + (double)tv.tv_usec / 1000000.0);
}

void printMatrix(const float* matrix, int rows, int cols, const char* name) {
    printf("Matrix %s:\n", name);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%8.4f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
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
    cl_device_id deviceId = NULL;
    cl_context context;
    cl_command_queue commandQueue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;
	
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
        printf("Build error! Code: %d\n", err);
        size_t real_size;
        err = clGetProgramBuildInfo(
            program,
            deviceId,
            CL_PROGRAM_BUILD_LOG,
            0,
            NULL,
            &real_size
        );
        char* build_log = (char*)malloc(sizeof(char) * (real_size + 1));
        err = clGetProgramBuildInfo(
            program,
            deviceId,
            CL_PROGRAM_BUILD_LOG,
            real_size + 1,
            build_log,
            &real_size
        );
        // build_log[real_size] = 0;
        printf("Real size : %d\n", real_size);
        printf("Build log : %s\n", build_log);
        free(build_log);
        return EXIT_FAILURE;
    }
	
	// Create kernel object
    kernel = clCreateKernel(program, "matrixMultip", &err);
    if (!kernel || err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create kernel object.\n");
        return EXIT_FAILURE;
    }
	
	const int M = 2000;
	const int N = 2000;
	const int K = 2000;
	
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
	
	double startTimeSeq = getTimeStamp();
	
	matrixMultip(A, B, D, M, N, K);
	
	double endTimeSeq = getTimeStamp();
	printf("Sequential execution time: %.9f seconds\n", endTimeSeq - startTimeSeq);
	
	err = clEnqueueWriteBuffer(commandQueue, bufferA, CL_TRUE, 0, sizeof(float) * M * K, A, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commandQueue, bufferB, CL_TRUE, 0, sizeof(float) * K * N, B, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to write data to device.\n");
        return EXIT_FAILURE;
    }
	
    fflush(stdout);
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bufferA);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufferB);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bufferC);
    err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&M);
    err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&N);
    err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&K);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to set kernel arguments.\n");
        return EXIT_FAILURE;
    }
	
	double startTimeCL = getTimeStamp();
	
	// Execute the kernel
	size_t globalWorkSize[2] = {M, N};
	err = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to enqueue kernel.\n");
		return EXIT_FAILURE;
	}
	
	double endTimeCL = getTimeStamp();
    printf("OpenCL execution time: %.9f seconds\n", endTimeCL - startTimeCL);
	
	// Read the result bufferC into host array C
	err = clEnqueueReadBuffer(commandQueue, bufferC, CL_TRUE, 0, sizeof(float) * M * N, C, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to read buffer C.\n");
		return EXIT_FAILURE;
	}
	
	//printMatrix(A, M, K, "A");
	//printMatrix(B, K, N, "B");
	//printMatrix(C, M, N, "C (OpenCL)");
	//printMatrix(D, M, N, "D (Sequential");
	
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
    free(D);

    return EXIT_SUCCESS
	
;}
