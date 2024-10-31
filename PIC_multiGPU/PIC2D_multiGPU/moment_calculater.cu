#include "moment_calculater.hpp"


MomentCalculater::MomentCalculater(MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo)
{
}


void MomentCalculater::resetZerothMomentOfOneSpecies(
    thrust::device_vector<ZerothMoment>& zerothMomentOfOneSpecies
)
{
    thrust::fill(
        zerothMomentOfOneSpecies.begin(), 
        zerothMomentOfOneSpecies.end(), 
        ZerothMoment()
    );
    cudaDeviceSynchronize();
}

void MomentCalculater::resetFirstMomentOfOneSpecies(
    thrust::device_vector<FirstMoment>& firstMomentOfOneSpecies
)
{
    thrust::fill(
        firstMomentOfOneSpecies.begin(), 
        firstMomentOfOneSpecies.end(), 
        FirstMoment()
    );
    cudaDeviceSynchronize();
}

void MomentCalculater::resetSecondMomentOfOneSpecies(
    thrust::device_vector<SecondMoment>& secondMomentOfOneSpecies
)
{
    thrust::fill(
        secondMomentOfOneSpecies.begin(), 
        secondMomentOfOneSpecies.end(), 
        SecondMoment()
    );
    cudaDeviceSynchronize();
}

//////////

__global__ void calculateZerothMomentOfOneSpecies_kernel(
    ZerothMoment* zerothMomentOfOneSpecies, 
    const Particle* particlesSpecies, 
    const unsigned long long existNumSpecies, 
    const int localNx, const int localNy, const int buffer, 
    const int localSizeX, const int localSizeY, 
    const float xminForProcs, const float xmaxForProcs, 
    const float yminForProcs, const float ymaxForProcs
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {

        float cx1, cx2; 
        int xIndex1, xIndex2;
        float xOverDx;
        float cy1, cy2; 
        int yIndex1, yIndex2;
        float yOverDy;

        xOverDx = (particlesSpecies[i].x - xminForProcs + buffer * device_dx) / device_dx;
        yOverDy = (particlesSpecies[i].y - yminForProcs + buffer * device_dy) / device_dy;

        xIndex1 = floorf(xOverDx);
        xIndex2 = xIndex1 + 1;
        xIndex2 = (xIndex2 == localSizeX) ? 0 : xIndex2;
        yIndex1 = floorf(yOverDy);
        yIndex2 = yIndex1 + 1;
        yIndex2 = (yIndex2 == localSizeY) ? 0 : yIndex2;
        if (xIndex1 < 0 || xIndex1 >= localSizeX) return;
        if (yIndex1 < 0 || yIndex1 >= localSizeY) return;

        cx1 = xOverDx - xIndex1;
        cx2 = 1.0f - cx1;
        cy1 = yOverDy - yIndex1;
        cy2 = 1.0f - cy1;

        atomicAdd(&(zerothMomentOfOneSpecies[yIndex1 + localSizeY * xIndex1].n), cx2 * cy2);
        atomicAdd(&(zerothMomentOfOneSpecies[yIndex2 + localSizeY * xIndex1].n), cx2 * cy1);
        atomicAdd(&(zerothMomentOfOneSpecies[yIndex1 + localSizeY * xIndex2].n), cx1 * cy2);
        atomicAdd(&(zerothMomentOfOneSpecies[yIndex2 + localSizeY * xIndex2].n), cx1 * cy1);
    }
};


void MomentCalculater::calculateZerothMomentOfOneSpecies(
    thrust::device_vector<ZerothMoment>& zerothMomentOfOneSpecies, 
    const thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long existNumSpecies
)
{
    resetZerothMomentOfOneSpecies(zerothMomentOfOneSpecies);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    calculateZerothMomentOfOneSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(zerothMomentOfOneSpecies.data()), 
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        existNumSpecies, 
        mPIInfo.localNx, mPIInfo.localNy, mPIInfo.buffer, 
        mPIInfo.localSizeX, mPIInfo.localSizeY, 
        xminForProcs, xmaxForProcs, yminForProcs, ymaxForProcs
    );
    cudaDeviceSynchronize();
}



__global__ void calculateFirstMomentOfOneSpecies_kernel(
    FirstMoment* firstMomentOfOneSpecies, 
    const Particle* particlesSpecies, 
    const unsigned long long existNumSpecies, 
    const int localNx, const int localNy, const int buffer, 
    const int localSizeX, const int localSizeY, 
    const float xminForProcs, const float xmaxForProcs, 
    const float yminForProcs, const float ymaxForProcs
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
    
        float cx1, cx2; 
        int xIndex1, xIndex2;
        float xOverDx;
        float cy1, cy2; 
        int yIndex1, yIndex2;
        float yOverDy;
        float vx, vy, vz;

        xOverDx = (particlesSpecies[i].x - xminForProcs + buffer * device_dx) / device_dx;
        yOverDy = (particlesSpecies[i].y - yminForProcs + buffer * device_dy) / device_dy;

        xIndex1 = floorf(xOverDx);
        xIndex2 = xIndex1 + 1;
        xIndex2 = (xIndex2 == localSizeX) ? 0 : xIndex2;
        yIndex1 = floorf(yOverDy);
        yIndex2 = yIndex1 + 1;
        yIndex2 = (yIndex2 == localSizeY) ? 0 : yIndex2;
        if (xIndex1 < 0 || xIndex1 >= localSizeX) return;
        if (yIndex1 < 0 || yIndex1 >= localSizeY) return;

        cx1 = xOverDx - xIndex1;
        cx2 = 1.0f - cx1;
        cy1 = yOverDy - yIndex1;
        cy2 = 1.0f - cy1;

        vx = particlesSpecies[i].vx / particlesSpecies[i].gamma;
        vy = particlesSpecies[i].vy / particlesSpecies[i].gamma;
        vz = particlesSpecies[i].vz / particlesSpecies[i].gamma;

        atomicAdd(&(firstMomentOfOneSpecies[yIndex1 + localSizeY * xIndex1].x), vx * cx2 * cy2);
        atomicAdd(&(firstMomentOfOneSpecies[yIndex2 + localSizeY * xIndex1].x), vx * cx2 * cy1);
        atomicAdd(&(firstMomentOfOneSpecies[yIndex1 + localSizeY * xIndex2].x), vx * cx1 * cy2);
        atomicAdd(&(firstMomentOfOneSpecies[yIndex2 + localSizeY * xIndex2].x), vx * cx1 * cy1);

        atomicAdd(&(firstMomentOfOneSpecies[yIndex1 + localSizeY * xIndex1].y), vy * cx2 * cy2);
        atomicAdd(&(firstMomentOfOneSpecies[yIndex2 + localSizeY * xIndex1].y), vy * cx2 * cy1);
        atomicAdd(&(firstMomentOfOneSpecies[yIndex1 + localSizeY * xIndex2].y), vy * cx1 * cy2);
        atomicAdd(&(firstMomentOfOneSpecies[yIndex2 + localSizeY * xIndex2].y), vy * cx1 * cy1);

        atomicAdd(&(firstMomentOfOneSpecies[yIndex1 + localSizeY * xIndex1].z), vz * cx2 * cy2);
        atomicAdd(&(firstMomentOfOneSpecies[yIndex2 + localSizeY * xIndex1].z), vz * cx2 * cy1);
        atomicAdd(&(firstMomentOfOneSpecies[yIndex1 + localSizeY * xIndex2].z), vz * cx1 * cy2);
        atomicAdd(&(firstMomentOfOneSpecies[yIndex2 + localSizeY * xIndex2].z), vz * cx1 * cy1);
    }
};


void MomentCalculater::calculateFirstMomentOfOneSpecies(
    thrust::device_vector<FirstMoment>& firstMomentOfOneSpecies, 
    const thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long existNumSpecies
)
{
    resetFirstMomentOfOneSpecies(firstMomentOfOneSpecies);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    calculateFirstMomentOfOneSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(firstMomentOfOneSpecies.data()), 
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        existNumSpecies, 
        mPIInfo.localNx, mPIInfo.localNy, mPIInfo.buffer, 
        mPIInfo.localSizeX, mPIInfo.localSizeY, 
        xminForProcs, xmaxForProcs, yminForProcs, ymaxForProcs
    );
    cudaDeviceSynchronize();
}


__global__ void calculateSecondMomentOfOneSpecies_kernel(
    SecondMoment* secondMomentOfOneSpecies, 
    const Particle* particlesSpecies, 
    const unsigned long long existNumSpecies, 
    const int localNx, const int localNy, const int buffer, 
    const int localSizeX, const int localSizeY, 
    const float xminForProcs, const float xmaxForProcs, 
    const float yminForProcs, const float ymaxForProcs
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
    
        float cx1, cx2; 
        int xIndex1, xIndex2;
        float xOverDx;
        float cy1, cy2; 
        int yIndex1, yIndex2;
        float yOverDy;
        float vx, vy, vz;

        xOverDx = (particlesSpecies[i].x - xminForProcs + buffer * device_dx) / device_dx;
        yOverDy = (particlesSpecies[i].y - yminForProcs + buffer * device_dy) / device_dy;

        xIndex1 = floorf(xOverDx);
        xIndex2 = xIndex1 + 1;
        xIndex2 = (xIndex2 == localSizeX) ? 0 : xIndex2;
        yIndex1 = floorf(yOverDy);
        yIndex2 = yIndex1 + 1;
        yIndex2 = (yIndex2 == localSizeY) ? 0 : yIndex2;
        if (xIndex1 < 0 || xIndex1 >= localSizeX) return;
        if (yIndex1 < 0 || yIndex1 >= localSizeY) return;

        cx1 = xOverDx - xIndex1;
        cx2 = 1.0f - cx1;
        cy1 = yOverDy - yIndex1;
        cy2 = 1.0f - cy1;

        vx = particlesSpecies[i].vx / particlesSpecies[i].gamma;
        vy = particlesSpecies[i].vy / particlesSpecies[i].gamma;
        vz = particlesSpecies[i].vz / particlesSpecies[i].gamma;

        atomicAdd(&(secondMomentOfOneSpecies[yIndex1 + localSizeY * xIndex1].xx), vx * vx * cx2 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex2 + localSizeY * xIndex1].xx), vx * vx * cx2 * cy1);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex1 + localSizeY * xIndex2].xx), vx * vx * cx1 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex2 + localSizeY * xIndex2].xx), vx * vx * cx1 * cy1);

        atomicAdd(&(secondMomentOfOneSpecies[yIndex1 + localSizeY * xIndex1].yy), vy * vy * cx2 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex2 + localSizeY * xIndex1].yy), vy * vy * cx2 * cy1);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex1 + localSizeY * xIndex2].yy), vy * vy * cx1 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex2 + localSizeY * xIndex2].yy), vy * vy * cx1 * cy1);

        atomicAdd(&(secondMomentOfOneSpecies[yIndex1 + localSizeY * xIndex1].zz), vz * vz * cx2 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex2 + localSizeY * xIndex1].zz), vz * vz * cx2 * cy1);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex1 + localSizeY * xIndex2].zz), vz * vz * cx1 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex2 + localSizeY * xIndex2].zz), vz * vz * cx1 * cy1);

        atomicAdd(&(secondMomentOfOneSpecies[yIndex1 + localSizeY * xIndex1].xy), vx * vy * cx2 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex2 + localSizeY * xIndex1].xy), vx * vy * cx2 * cy1);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex1 + localSizeY * xIndex2].xy), vx * vy * cx1 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex2 + localSizeY * xIndex2].xy), vx * vy * cx1 * cy1);

        atomicAdd(&(secondMomentOfOneSpecies[yIndex1 + localSizeY * xIndex1].xz), vx * vz * cx2 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex2 + localSizeY * xIndex1].xz), vx * vz * cx2 * cy1);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex1 + localSizeY * xIndex2].xz), vx * vz * cx1 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex2 + localSizeY * xIndex2].xz), vx * vz * cx1 * cy1);

        atomicAdd(&(secondMomentOfOneSpecies[yIndex1 + localSizeY * xIndex1].yz), vy * vz * cx2 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex2 + localSizeY * xIndex1].yz), vy * vz * cx2 * cy1);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex1 + localSizeY * xIndex2].yz), vy * vz * cx1 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex2 + localSizeY * xIndex2].yz), vy * vz * cx1 * cy1);
    }
};


void MomentCalculater::calculateSecondMomentOfOneSpecies(
    thrust::device_vector<SecondMoment>& secondMomentOfOneSpecies, 
    const thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long existNumSpecies
)
{
    resetSecondMomentOfOneSpecies(secondMomentOfOneSpecies);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    calculateSecondMomentOfOneSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(secondMomentOfOneSpecies.data()), 
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        existNumSpecies, 
        mPIInfo.localNx, mPIInfo.localNy, mPIInfo.buffer, 
        mPIInfo.localSizeX, mPIInfo.localSizeY, 
        xminForProcs, xmaxForProcs, yminForProcs, ymaxForProcs
    );
    cudaDeviceSynchronize();
}



