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
    double xminForProcs = xmin + (xmax - xmin) / mPIInfo.procs * mPIInfo.rank;
    double xmaxForProcs = xmin + (xmax - xmin) / mPIInfo.procs * (mPIInfo.rank + 1);

    calculateCurrentOfOneSpecies(
        current, particlesIon, qIon, 
        mPIInfo.existNumIonPerProcs, 
        xminForProcs, xmaxForProcs
    );
    calculateCurrentOfOneSpecies(
        current, particlesElectron, qElectron, 
        mPIInfo.existNumElectronPerProcs, 
        xminForProcs, xmaxForProcs
    );
}


__global__ void calculateCurrentOfOneSpecies_kernel(
    CurrentField* current, const Particle* particlesSpecies, 
    const double q, const int existNumSpecies, 
    const double xminForProcs, const double xmaxForProcs
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        double cx1, cx2; 
        int xIndex1, xIndex2;
        double xOverDx;
        double qOverGamma, qVxOverGamma, qVyOverGamma, qVzOverGamma;

        xOverDx = (particlesSpecies[i].x - xminForProcs + device_dx) / device_dx;

        xIndex1 = floorf(xOverDx);
        xIndex2 = xIndex1 + 1;
        
        cx1 = xOverDx - xIndex1;
        cx2 = 1.0 - cx1;

        qOverGamma = q / particlesSpecies[i].gamma;
        qVxOverGamma = qOverGamma * particlesSpecies[i].vx;
        qVyOverGamma = qOverGamma * particlesSpecies[i].vy;
        qVzOverGamma = qOverGamma * particlesSpecies[i].vz;

        atomicAdd(&(current[xIndex1].jX), qVxOverGamma * cx2);
        atomicAdd(&(current[xIndex2].jX), qVxOverGamma * cx1);

        atomicAdd(&(current[xIndex1].jY), qVyOverGamma * cx2);
        atomicAdd(&(current[xIndex2].jY), qVyOverGamma * cx1);

        atomicAdd(&(current[xIndex1].jZ), qVzOverGamma * cx2);
        atomicAdd(&(current[xIndex2].jZ), qVzOverGamma * cx1);
    }
};


void CurrentCalculator::calculateCurrentOfOneSpecies(
    thrust::device_vector<CurrentField>& current, 
    const thrust::device_vector<Particle>& particlesSpecies, 
    const double q, const int existNumSpecies, 
    const double xminForProcs, const double xmaxForProcs
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    calculateCurrentOfOneSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        q, existNumSpecies, 
        xminForProcs, xmaxForProcs
    );
}


