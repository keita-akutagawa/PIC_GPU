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
    const thrust::device_vector<Particle>& particlesElectron, 
    const thrust::device_vector<unsigned long long>& particleIndexIon, 
    const thrust::device_vector<unsigned long long>& particleIndexElectron
)
{
    calculateCurrentOfOneSpecies(
        current, particlesIon, particleIndexIon, qIon, totalNumIon
    );
    calculateCurrentOfOneSpecies(
        current, particlesElectron, particleIndexElectron, qElectron, totalNumElectron
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

__device__ void cudaAssertInt(bool condition, const int message) {
    if (!condition) {
        printf("Assertion failed: %d\n", message);
    }
}

__global__ void calculateCurrentOfOneSpecies_kernel(
    CurrentField* current,
    const Particle* particlesSpecies, 
    const float q, const int totalNumSpecies
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    const int invalidIndex = -1;
    extern __shared__ IndexCurrent sharedCurrent1[];
    extern __shared__ IndexCurrent sharedCurrent2[];
    extern __shared__ IndexCurrent sharedCurrent3[];
    extern __shared__ IndexCurrent sharedCurrent4[];


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

        sharedCurrent1[tid].index = yIndex1 + device_ny * xIndex1;
        sharedCurrent2[tid].index = yIndex2 + device_ny * xIndex1;
        sharedCurrent3[tid].index = yIndex1 + device_ny * xIndex2;
        sharedCurrent4[tid].index = yIndex2 + device_ny * xIndex2;

        sharedCurrent1[tid].jX = qVxOverGamma * cx2 * cy2;
        sharedCurrent2[tid].jX = qVxOverGamma * cx2 * cy1;
        sharedCurrent3[tid].jX = qVxOverGamma * cx1 * cy2;
        sharedCurrent4[tid].jX = qVxOverGamma * cx1 * cy1;

        sharedCurrent1[tid].jY = qVyOverGamma * cx2 * cy2;
        sharedCurrent2[tid].jY = qVyOverGamma * cx2 * cy1;
        sharedCurrent3[tid].jY = qVyOverGamma * cx1 * cy2;
        sharedCurrent4[tid].jY = qVyOverGamma * cx1 * cy1;

        sharedCurrent1[tid].jZ = qVzOverGamma * cx2 * cy2;
        sharedCurrent2[tid].jZ = qVzOverGamma * cx2 * cy1;
        sharedCurrent3[tid].jZ = qVzOverGamma * cx1 * cy2;
        sharedCurrent4[tid].jZ = qVzOverGamma * cx1 * cy1;
    } 

    __syncthreads();


    /*        
    for (int reductionCount = 1; reductionCount < min(blockDim.x, 3); reductionCount *= 2) {
        if (tid % (2 * reductionCount) == 0) {
            int index1, index2;

            index1 = sharedCurrent1[tid].index;
            index2 = sharedCurrent1[tid + reductionCount].index;
            if (index1 == index2) {
                sharedCurrent1[tid].jX += sharedCurrent1[tid + reductionCount].jX;
                sharedCurrent1[tid].jY += sharedCurrent1[tid + reductionCount].jY;
                sharedCurrent1[tid].jZ += sharedCurrent1[tid + reductionCount].jZ;
                sharedCurrent1[tid + reductionCount].index = invalidIndex;
            }

            index1 = sharedCurrent2[tid].index;
            index2 = sharedCurrent2[tid + reductionCount].index;
            if (index1 == index2) {
                sharedCurrent2[tid].jX += sharedCurrent2[tid + reductionCount].jX;
                sharedCurrent2[tid].jY += sharedCurrent2[tid + reductionCount].jY;
                sharedCurrent2[tid].jZ += sharedCurrent2[tid + reductionCount].jZ;
                sharedCurrent2[tid + reductionCount].index = invalidIndex;
            }

            index1 = sharedCurrent3[tid].index;
            index2 = sharedCurrent3[tid + reductionCount].index;
            if (index1 == index2) {
                sharedCurrent3[tid].jX += sharedCurrent3[tid + reductionCount].jX;
                sharedCurrent3[tid].jY += sharedCurrent3[tid + reductionCount].jY;
                sharedCurrent3[tid].jZ += sharedCurrent3[tid + reductionCount].jZ;
                sharedCurrent3[tid + reductionCount].index = invalidIndex;
            }

            index1 = sharedCurrent4[tid].index;
            index2 = sharedCurrent4[tid + reductionCount].index;
            if (index1 == index2) {
                sharedCurrent4[tid].jX += sharedCurrent4[tid + reductionCount].jX;
                sharedCurrent4[tid].jY += sharedCurrent4[tid + reductionCount].jY;
                sharedCurrent4[tid].jZ += sharedCurrent4[tid + reductionCount].jZ;
                sharedCurrent4[tid + reductionCount].index = invalidIndex;
            }
            
        }
        __syncthreads();
    }
    */


    if (sharedCurrent1[tid].index != invalidIndex) {
        int index;
        float jX, jY, jZ;

        index = sharedCurrent1[tid].index;
        jX = sharedCurrent1[tid].jX;
        jY = sharedCurrent1[tid].jY;
        jZ = sharedCurrent1[tid].jZ;
        atomicAdd(&(current[index].jX), jX);
        atomicAdd(&(current[index].jY), jY);
        atomicAdd(&(current[index].jZ), jZ);

        sharedCurrent1[tid].initialize();
    }

    if (sharedCurrent2[tid].index != invalidIndex) {
        int index;
        float jX, jY, jZ;

        index = sharedCurrent2[tid].index;
        jX = sharedCurrent2[tid].jX;
        jY = sharedCurrent2[tid].jY;
        jZ = sharedCurrent2[tid].jZ;
        atomicAdd(&(current[index].jX), jX);
        atomicAdd(&(current[index].jY), jY);
        atomicAdd(&(current[index].jZ), jZ);

        sharedCurrent2[tid].initialize();
    }

    if (sharedCurrent3[tid].index != invalidIndex) {
        int index;
        float jX, jY, jZ;

        index = sharedCurrent3[tid].index;
        jX = sharedCurrent3[tid].jX;
        jY = sharedCurrent3[tid].jY;
        jZ = sharedCurrent3[tid].jZ;
        atomicAdd(&(current[index].jX), jX);
        atomicAdd(&(current[index].jY), jY);
        atomicAdd(&(current[index].jZ), jZ);

        sharedCurrent3[tid].initialize();
    }

    if (sharedCurrent4[tid].index != invalidIndex) {
        int index;
        float jX, jY, jZ;

        index = sharedCurrent4[tid].index;
        jX = sharedCurrent4[tid].jX;
        jY = sharedCurrent4[tid].jY;
        jZ = sharedCurrent4[tid].jZ;
        atomicAdd(&(current[index].jX), jX);
        atomicAdd(&(current[index].jY), jY);
        atomicAdd(&(current[index].jZ), jZ);

        sharedCurrent4[tid].initialize();
    }
    __syncthreads();

};


void CurrentCalculator::calculateCurrentOfOneSpecies(
    thrust::device_vector<CurrentField>& current, 
    const thrust::device_vector<Particle>& particlesSpecies, 
    const thrust::device_vector<unsigned long long>& particlesIndexSpecies, 
    float q, int totalNumSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((totalNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);
    int sharedMemorySize = 256 * sizeof(IndexCurrent);

    calculateCurrentOfOneSpecies_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemorySize>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        q, totalNumSpecies
    );

    cudaDeviceSynchronize();
}


