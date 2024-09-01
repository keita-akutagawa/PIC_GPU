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
    MPIInfo& mPIInfo
)
{
    calculateCurrentOfOneSpecies(
        current, particlesIon, qIon, 
        mPIInfo.existNumIonPerProcs, mPIInfo.localNx, 
        xmin + (xmax - xmin) / mPIInfo.procs * mPIInfo.rank, xmin + (xmax - xmin) / mPIInfo.procs * (mPIInfo.rank + 1)
    );
    calculateCurrentOfOneSpecies(
        current, particlesElectron, qElectron, 
        mPIInfo.existNumElectronPerProcs, mPIInfo.localNx, 
        xmin + (xmax - xmin) / mPIInfo.procs * mPIInfo.rank, xmin + (xmax - xmin) / mPIInfo.procs * (mPIInfo.rank + 1)
    );
}


// 必ず見直すこと。インデックスが恐らく間違っている
// -1 ~ localNx + 1 なのか、0 ~ localNx + 2 なのか
__global__ void calculateCurrentOfOneSpecies_kernel(
    CurrentField* current, const Particle* particlesSpecies, 
    const float q, const int existNumSpecies, int localNx, 
    const float xminForProcs, const float xmaxForProcs
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        float cx1, cx2; 
        int xIndex1, xIndex2;
        float xOverDx;
        float qOverGamma, qVxOverGamma, qVyOverGamma, qVzOverGamma;

        xOverDx = (particlesSpecies[i].x - xminForProcs) / device_dx;

        xIndex1 = floorf(xOverDx);
        xIndex1 == (xIndex1 == -1) ? 0 : xIndex1;
        xIndex2 = xIndex1 + 1;
        xIndex2 = (xIndex2 == localNx) ? 0 : xIndex2;

        cx1 = xOverDx - xIndex1;
        cx2 = 1.0f - cx1;

        qOverGamma = q / particlesSpecies[i].gamma;
        qVxOverGamma = qOverGamma * particlesSpecies[i].vx;
        qVyOverGamma = qOverGamma * particlesSpecies[i].vy;
        qVzOverGamma = qOverGamma * particlesSpecies[i].vz;

        atomicAdd(&(current[xIndex1].jX), qVxOverGamma * cx2 * min(1, xIndex1));
        atomicAdd(&(current[xIndex2].jX), qVxOverGamma * cx1 * min(1, xIndex2));

        atomicAdd(&(current[xIndex1].jY), qVyOverGamma * cx2 * min(1, xIndex1));
        atomicAdd(&(current[xIndex2].jY), qVyOverGamma * cx1 * min(1, xIndex2));

        atomicAdd(&(current[xIndex1].jZ), qVzOverGamma * cx2 * min(1, xIndex1));
        atomicAdd(&(current[xIndex2].jZ), qVzOverGamma * cx1 * min(1, xIndex2));
    }
};


void CurrentCalculator::calculateCurrentOfOneSpecies(
    thrust::device_vector<CurrentField>& current, 
    const thrust::device_vector<Particle>& particlesSpecies, 
    const float q, const int existNumSpecies, const int localNx, 
    const float xminForProcs, const float xmaxForProcs
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    calculateCurrentOfOneSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        q, existNumSpecies, localNx, 
        xminForProcs, xmaxForProcs
    );
}


