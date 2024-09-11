#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <string>

#define CHECK_ERROR(err, msg) \
    if (err != CL_SUCCESS) { \
        std::cerr << msg << ": " << err << std::endl; \
        exit(EXIT_FAILURE); \
    }

const char* kernelSource = R"CLC(
__kernel void knapsack(
    __global const int* capacities,   // Input: array of capacities (W)
    __global const int* weights,      // Input: array of item weights
    __global const int* values,       // Input: array of item values
    __global int* results,            // Output: array for results
    const int numItems,               // Number of items
    const int maxCapacity             // Maximum capacity for the knapsack
) {
    int idx = get_global_id(0); // Each thread works on a different knapsack problem (by index)
    
    __local int dp[101]; // Assuming maxCapacity <= 100

    for (int i = 0; i <= maxCapacity; i++) {
        dp[i] = 0;
    }
    
    for (int i = 0; i < numItems; i++) {
        int w = weights[idx * numItems + i]; // Weight of item i for this problem
        int v = values[idx * numItems + i];  // Value of item i for this problem

        for (int cap = maxCapacity; cap >= w; cap--) {
            dp[cap] = max(dp[cap], dp[cap - w] + v);
        }
    }

    results[idx] = dp[maxCapacity];
}
)CLC";

void generateRandomTestCases(int numProblems, int numItems, std::vector<int>& capacities, 
                             std::vector<int>& weights, std::vector<int>& values, 
                             int maxCapacity, int maxWeight, int maxValue) {
    // Random seed
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> capDist(1, maxCapacity);  
    std::uniform_int_distribution<> weightDist(1, maxWeight); 
    std::uniform_int_distribution<> valueDist(1, maxValue);   

    for (int i = 0; i < numProblems; i++) {
        capacities.push_back(capDist(gen));
    }

    for (int i = 0; i < numProblems; i++) {
        for (int j = 0; j < numItems; j++) {
            weights.push_back(weightDist(gen));
            values.push_back(valueDist(gen));
        }
    }
}

int main() {
    cl_int err;
    cl_uint numPlatforms;
    cl_platform_id platform = NULL;
    err = clGetPlatformIDs(1, &platform, &numPlatforms);
    CHECK_ERROR(err, "Failed to find platform");

    cl_uint numDevices;
    cl_device_id device = NULL;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &numDevices);
    CHECK_ERROR(err, "Failed to find GPU device");

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err, "Failed to create context");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err, "Failed to create command queue");

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    CHECK_ERROR(err, "Failed to create program");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    CHECK_ERROR(err, "Failed to build program");

    cl_kernel kernel = clCreateKernel(program, "knapsack", &err);
    CHECK_ERROR(err, "Failed to create kernel");

    int numProblems = 100;   
    int numItems = 10;       
    int maxCapacity = 100;  
    int maxWeight = 50;      
    int maxValue = 200;      

    std::vector<int> capacities;
    std::vector<int> weights;
    std::vector<int> values;
    generateRandomTestCases(numProblems, numItems, capacities, weights, values, maxCapacity, maxWeight, maxValue);

    std::vector<int> results(numProblems, 0);

    cl_mem d_capacities = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * numProblems, capacities.data(), &err);
    CHECK_ERROR(err, "Failed to create buffer for capacities");

    cl_mem d_weights = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * numProblems * numItems, weights.data(), &err);
    CHECK_ERROR(err, "Failed to create buffer for weights");

    cl_mem d_values = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * numProblems * numItems, values.data(), &err);
    CHECK_ERROR(err, "Failed to create buffer for values");

    cl_mem d_results = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * numProblems, NULL, &err);
    CHECK_ERROR(err, "Failed to create buffer for results");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_capacities);
    CHECK_ERROR(err, "Failed to set kernel argument 0");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_weights);
    CHECK_ERROR(err, "Failed to set kernel argument 1");

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_values);
    CHECK_ERROR(err, "Failed to set kernel argument 2");

    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_results);
    CHECK_ERROR(err, "Failed to set kernel argument 3");

    err = clSetKernelArg(kernel, 4, sizeof(int), &numItems);
    CHECK_ERROR(err, "Failed to set kernel argument 4");

    err = clSetKernelArg(kernel, 5, sizeof(int), &maxCapacity);
    CHECK_ERROR(err, "Failed to set kernel argument 5");

    size_t globalWorkSize = numProblems;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);
    CHECK_ERROR(err, "Failed to enqueue kernel");

    clFinish(queue);

    err = clEnqueueReadBuffer(queue, d_results, CL_TRUE, 0, sizeof(int) * numProblems, results.data(), 0, NULL, NULL);
    CHECK_ERROR(err, "Failed to read results");

    for (int i = 0; i < numProblems; i++) {
        std::cout << "Knapsack problem " << i + 1 << ": Maximum value = " << results[i] << std::endl;
    }

    clReleaseMemObject(d_capacities);
    clReleaseMemObject(d_weights);
    clReleaseMemObject(d_values);
    clReleaseMemObject(d_results);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
