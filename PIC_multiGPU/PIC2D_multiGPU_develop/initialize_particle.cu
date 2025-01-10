#include "initialize_particle.hpp"
#include <thrust/transform.h>
#include <thrust/random.h>
#include <curand_kernel.h>
#include <cmath>
#include <random>


InitializeParticle::InitializeParticle(MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo)
{
}


__global__ void uniformForPosition_x_kernel(
    Particle* particle, 
    const unsigned long long nStart, const unsigned long long nEnd, 
    const float xmin, const float xmax, 
    const int seed, const unsigned long long offset
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, offset + i, 0, &state);
        float x = curand_uniform(&state) * (xmax - xmin) + xmin;
        particle[i + nStart].x = x;
        particle[i + nStart].isExist = true;
    }
}

void InitializeParticle::uniformForPosition_x(
    unsigned long long nStart, 
    unsigned long long nEnd, 
    float xmin, float xmax, 
    int seed, 
    thrust::device_vector<Particle>& particlesSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nEnd - nStart + threadsPerBlock.x - 1) / threadsPerBlock.x);

    uniformForPosition_x_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        nStart, nEnd,
        xmin, xmax, 
        seed, mPIInfo.rank * (nEnd - nStart)
    );
    cudaDeviceSynchronize();
}


__global__ void uniformForPosition_y_kernel(
    Particle* particle, 
    const unsigned long long nStart, const unsigned long long nEnd, 
    const float ymin, const float ymax, 
    const int seed, const unsigned long long offset
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, offset + i, 0, &state);
        float y = curand_uniform(&state) * (ymax - ymin) + ymin;
        particle[i + nStart].y = y;
        particle[i + nStart].isExist = true;
    }
}

void InitializeParticle::uniformForPosition_y(
    unsigned long long nStart, 
    unsigned long long nEnd, 
    float ymin, float ymax, 
    int seed, 
    thrust::device_vector<Particle>& particlesSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nEnd - nStart + threadsPerBlock.x - 1) / threadsPerBlock.x);

    uniformForPosition_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        nStart, nEnd,
        ymin, ymax, 
        seed, mPIInfo.rank * (nEnd - nStart)
    );
    cudaDeviceSynchronize();
}

//////////

__global__ void maxwellDistributionForVelocity_kernel(
    Particle* particle, 
    const float bulkVxSpecies, const float bulkVySpecies, const float bulkVzSpecies, 
    const float vxThSpecies, const float vyThSpecies, const float vzThSpecies, 
    const unsigned long long nStart, const unsigned long long nEnd, 
    const int seed, const unsigned long long offset
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState stateVx; 
        curandState stateVy; 
        curandState stateVz; 
        curand_init(seed,           100 * (offset + i), 0, &stateVx);
        curand_init(seed + 1000000, 100 * (offset + i), 0, &stateVy);
        curand_init(seed + 2000000, 100 * (offset + i), 0, &stateVz);

        float vx, vy, vz, gamma;

        while (true) {
            vx = bulkVxSpecies + curand_normal(&stateVx) * vxThSpecies;
            vy = bulkVySpecies + curand_normal(&stateVy) * vyThSpecies;
            vz = bulkVzSpecies + curand_normal(&stateVz) * vzThSpecies;

            if (vx * vx + vy * vy + vz * vz < device_c * device_c) break;
        }

        gamma = 1.0f / sqrt(1.0f - (vx * vx + vy * vy + vz * vz) / (device_c * device_c));

        particle[i + nStart].vx = vx * gamma;
        particle[i + nStart].vy = vy * gamma;
        particle[i + nStart].vz = vz * gamma;
        particle[i + nStart].gamma = gamma;
    }
}


void InitializeParticle::maxwellDistributionForVelocity(
    float bulkVxSpecies, 
    float bulkVySpecies, 
    float bulkVzSpecies, 
    float vxThSpecies, 
    float vyThSpecies, 
    float vzThSpecies, 
    unsigned long long nStart, 
    unsigned long long nEnd, 
    int seed, 
    thrust::device_vector<Particle>& particlesSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nEnd - nStart + threadsPerBlock.x - 1) / threadsPerBlock.x);

    maxwellDistributionForVelocity_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        bulkVxSpecies, bulkVySpecies, bulkVzSpecies, 
        vxThSpecies, vyThSpecies, vzThSpecies, 
        nStart, nEnd, seed, mPIInfo.rank * (nEnd - nStart)
    );
    cudaDeviceSynchronize();
}


/*
__global__ void harrisForPositionY_kernel(
    Particle* particle, float sheatThickness, 
    const unsigned long long nStart, const unsigned long long nEnd, const int seed
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, 10 * i, 0, &state);
        float yCenter = 0.5f * (device_ymax - device_ymin) + device_ymin;

        float randomValue;
        float y;
        while (true) {
            randomValue = curand_uniform(&state);
            y = yCenter + sheatThickness * atanh(2.0f * randomValue - 1.0f);

            if (device_ymin < y && y < device_ymax) break;
        }
        
        particle[i + nStart].y = y;
    }
}

void InitializeParticle::harrisForPositionY(
    unsigned long long nStart, 
    unsigned long long nEnd, 
    int seed, 
    float sheatThickness, 
    thrust::device_vector<Particle>& particlesSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nEnd - nStart + threadsPerBlock.x - 1) / threadsPerBlock.x);

    harrisForPositionY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), sheatThickness, 
        nStart, nEnd, seed
    );

    cudaDeviceSynchronize();
}


__global__ void harrisBackgroundForPositionY_kernel(
    Particle* particle, float sheatThickness, 
    const unsigned long long nStart, const unsigned long long nEnd, const int seed
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, 10 * i, 0, &state);
        float yCenter = 0.5f * (device_ymax - device_ymin) + device_ymin;

        float randomValue;
        float y;
        while (true) {
            randomValue = curand_uniform(&state);
            y = randomValue * (device_ymax - device_ymin);

            if (randomValue < (1.0f - 1.0f / cosh((y - yCenter) / sheatThickness))) break;
        } 
        
        particle[i + nStart].y = y;
    }
}

void InitializeParticle::harrisBackgroundForPositionY(
    unsigned long long nStart, 
    unsigned long long nEnd, 
    int seed, 
    float sheatThickness, 
    thrust::device_vector<Particle>& particlesSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nEnd - nStart + threadsPerBlock.x - 1) / threadsPerBlock.x);

    harrisBackgroundForPositionY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), sheatThickness, 
        nStart, nEnd, seed
    );

    cudaDeviceSynchronize();
}
*/

