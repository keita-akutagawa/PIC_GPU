#include <cmath>
#include "current_calculator.hpp"
#include <thrust/fill.h>


void CurrentCalculator::resetCurrent(
    thrust::device_vector<CurrentField>& current
)
{
    thrust::fill(current.begin(), current.end(), CurrentField());
}


void CurrentCalculator::calculateCurrent(
    thrust::device_vector<CurrentField>& current, 
    const thrust::device_vector<Particle>& particlesIon, 
    const thrust::device_vector<Particle>& particlesElectron
)
{
    calculateCurrentOfOneSpecies(
        current, particlesIon, qIon, totalNumIon
    );
    calculateCurrentOfOneSpecies(
        current, particlesElectron, qElectron, totalNumElectron
    );
}


struct IndexCurrent
{
    int index;
    float jX;
    float jY;
    float jZ;

    __device__
    IndexCurrent() :
        index(-1),
        jX(0.0f),
        jY(0.0f),
        jZ(0.0f)
        {}
    

    __device__
    void initialize()
    {
        index = -1;
        jX = 0.0f;
        jY = 0.0f;
        jZ = 0.0f;
    }
};


__device__ void cudaAssert(bool condition, const char *message) {
    if (!condition) {
        printf("Assertion failed: %s\n", message);
    }
}

__device__ void cudaAssertInt(bool condition, const int number) {
    if (!condition) {
        printf("Assertion failed: %d\n", number);
    }
}


__global__ void calculateCurrentOfOneSpecies_kernel(
    CurrentField* current,
    const Particle* particlesSpecies,
    const float q, const int totalNumSpecies
)
{
    extern __shared__ IndexCurrent sharedCurrent[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    const int invalidIndex = -1;

    if (i < totalNumSpecies) {
        float cx1, cx2;
        int xIndex1, xIndex2;
        float xOverDx;
        float cy1, cy2;
        int yIndex1, yIndex2;
        float yOverDy;
        float qOverGamma, qVxOverGamma, qVyOverGamma, qVzOverGamma;

        xOverDx = particlesSpecies[i].x / device_dx;
        yOverDy = particlesSpecies[i].y / device_dy;

        xIndex1 = floorf(xOverDx);
        xIndex2 = xIndex1 + 1;
        xIndex2 = (xIndex2 == device_nx) ? 0 : xIndex2;
        yIndex1 = floorf(yOverDy);
        yIndex2 = yIndex1 + 1;
        yIndex2 = (yIndex2 == device_ny) ? 0 : yIndex2;

        cx1 = xOverDx - xIndex1;
        cx2 = 1.0f - cx1;
        cy1 = yOverDy - yIndex1;
        cy2 = 1.0f - cy1;

        qOverGamma = q / particlesSpecies[i].gamma;
        qVxOverGamma = qOverGamma * particlesSpecies[i].vx;
        qVyOverGamma = qOverGamma * particlesSpecies[i].vy;
        qVzOverGamma = qOverGamma * particlesSpecies[i].vz;

        sharedCurrent[4 * tid + 0].index = yIndex1 + device_ny * xIndex1;
        sharedCurrent[4 * tid + 1].index = yIndex2 + device_ny * xIndex1;
        sharedCurrent[4 * tid + 2].index = yIndex1 + device_ny * xIndex2;
        sharedCurrent[4 * tid + 3].index = yIndex2 + device_ny * xIndex2;

        sharedCurrent[4 * tid + 0].jX = qVxOverGamma * cx2 * cy2;
        sharedCurrent[4 * tid + 1].jX = qVxOverGamma * cx2 * cy1;
        sharedCurrent[4 * tid + 2].jX = qVxOverGamma * cx1 * cy2;
        sharedCurrent[4 * tid + 3].jX = qVxOverGamma * cx1 * cy1;

        sharedCurrent[4 * tid + 0].jY = qVyOverGamma * cx2 * cy2;
        sharedCurrent[4 * tid + 1].jY = qVyOverGamma * cx2 * cy1;
        sharedCurrent[4 * tid + 2].jY = qVyOverGamma * cx1 * cy2;
        sharedCurrent[4 * tid + 3].jY = qVyOverGamma * cx1 * cy1;

        sharedCurrent[4 * tid + 0].jZ = qVzOverGamma * cx2 * cy2;
        sharedCurrent[4 * tid + 1].jZ = qVzOverGamma * cx2 * cy1;
        sharedCurrent[4 * tid + 2].jZ = qVzOverGamma * cx1 * cy2;
        sharedCurrent[4 * tid + 3].jZ = qVzOverGamma * cx1 * cy1;
    }
    __syncthreads();
    

    int reductionWidth = 8;
    if (tid % reductionWidth == 0 && tid < 256) {
        for (int k = 1; k < reductionWidth; k++) {
            if (sharedCurrent[4 * tid + 0].index == sharedCurrent[4 * (tid + k) + 0].index) {
                sharedCurrent[4 * tid + 0].jX += sharedCurrent[4 * (tid + k) + 0].jX;
                sharedCurrent[4 * tid + 0].jY += sharedCurrent[4 * (tid + k) + 0].jY;
                sharedCurrent[4 * tid + 0].jZ += sharedCurrent[4 * (tid + k) + 0].jZ;
                sharedCurrent[4 * (tid + k) + 0].index = invalidIndex;
            }
        }
    }
    __syncthreads();
    


    if (sharedCurrent[4 * tid + 0].index != invalidIndex) {
        for(int j = 0; j < 4; j++) {
            int validIndex = sharedCurrent[4 * tid + j].index;
            atomicAdd(&(current[validIndex].jX), sharedCurrent[4 * tid + j].jX);
            atomicAdd(&(current[validIndex].jY), sharedCurrent[4 * tid + j].jY);
            atomicAdd(&(current[validIndex].jZ), sharedCurrent[4 * tid + j].jZ);
        }
    }
    __syncthreads();

    for (int j = 0; j < 4; j++) {
        sharedCurrent[4 * tid + j].initialize();
    }
}



void CurrentCalculator::calculateCurrentOfOneSpecies(
    thrust::device_vector<CurrentField>& current, 
    const thrust::device_vector<Particle>& particlesSpecies, 
    float q, int totalNumSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((totalNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);
    int sharedMemorySize = threadsPerBlock.x * sizeof(IndexCurrent) * 4;

    calculateCurrentOfOneSpecies_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemorySize>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        q, totalNumSpecies
    );

    cudaDeviceSynchronize();
}


