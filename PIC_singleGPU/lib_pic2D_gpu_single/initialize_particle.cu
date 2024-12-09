#include "initialize_particle.hpp"
#include <thrust/transform.h>
#include <thrust/random.h>
#include <curand_kernel.h>
#include <cmath>
#include <random>


__global__ void uniformForPositionX_kernel(
    Particle* particle, 
    const unsigned long long nStart, const unsigned long long nEnd, const int seed
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, i, 0, &state);
        float x = curand_uniform(&state) * (device_xmax - device_xmin) + device_xmin;
        particle[i + nStart].x = x;
    }
}

void InitializeParticle::uniformForPositionX(
    unsigned long long nStart, 
    unsigned long long nEnd, 
    int seed, 
    thrust::device_vector<Particle>& particlesSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nEnd - nStart + threadsPerBlock.x - 1) / threadsPerBlock.x);

    uniformForPositionX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        nStart, nEnd, seed
    );

    cudaDeviceSynchronize();
}


__global__ void uniformForPositionY_kernel(
    Particle* particle, 
    const unsigned long long nStart, const unsigned long long nEnd, const int seed
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, i, 0, &state);
        float y = curand_uniform(&state) * (device_ymax - device_ymin) + device_ymin;
        particle[i + nStart].y = y;
    }
}

void InitializeParticle::uniformForPositionY(
    unsigned long long nStart, 
    unsigned long long nEnd, 
    int seed, 
    thrust::device_vector<Particle>& particlesSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nEnd - nStart + threadsPerBlock.x - 1) / threadsPerBlock.x);

    uniformForPositionY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        nStart, nEnd, seed
    );

    cudaDeviceSynchronize();
}


__global__ void uniformForPositionYDetail_kernel(
    Particle* particle, 
    const unsigned long long nStart, const unsigned long long nEnd, const int seed, const float ymin, const float ymax
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, i, 0, &state);
        float y = curand_uniform(&state) * (ymax - ymin) + ymin;
        particle[i + nStart].y = y;
    }
}

void InitializeParticle::uniformForPositionYDetail(
    unsigned long long nStart, 
    unsigned long long nEnd, 
    int seed, 
    float ymin, 
    float ymax, 
    thrust::device_vector<Particle>& particlesSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nEnd - nStart + threadsPerBlock.x - 1) / threadsPerBlock.x);

    uniformForPositionYDetail_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        nStart, nEnd, seed, ymin, ymax
    );

    cudaDeviceSynchronize();
}


__global__ void maxwellDistributionForVelocity_kernel(
    Particle* particle, 
    const float bulkVxSpecies, const float bulkVySpecies, const float bulkVzSpecies, 
    const float vxThSpecies, const float vyThSpecies, const float vzThSpecies, 
    const unsigned long long nStart, const unsigned long long nEnd, const int seed
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState stateVx; 
        curandState stateVy; 
        curandState stateVz; 
        curand_init(seed,           100 * i, 0, &stateVx);
        curand_init(seed + 1000000, 100 * i, 0, &stateVy);
        curand_init(seed + 2000000, 100 * i, 0, &stateVz);

        float vx, vy, vz;

        while (true) {
            vx = bulkVxSpecies + curand_normal(&stateVx) * vxThSpecies;
            vy = bulkVySpecies + curand_normal(&stateVy) * vyThSpecies;
            vz = bulkVzSpecies + curand_normal(&stateVz) * vzThSpecies;

            if (vx * vx + vy * vy + vz * vz < device_c * device_c) break;
        }

        particle[i + nStart].vx = vx;
        particle[i + nStart].vy = vy;
        particle[i + nStart].vz = vz;
        particle[i + nStart].gamma = sqrt(1.0f + (vx * vx + vy * vy + vz * vz) / (device_c * device_c));
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
        nStart, nEnd, seed
    );

    cudaDeviceSynchronize();
}


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


//rejection sampling 
__global__ void fadeevForPosition_kernel(
    Particle* particle, float sheatThickness, float coefFadeev, 
    const unsigned long long nStart, const unsigned long long nEnd, const int seed
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState stateX; 
        curandState stateY; 
        curandState stateYForSample; 
        curand_init(seed,           100 * i, 0, &stateX);
        curand_init(seed + 1000000, 100 * i, 0, &stateY);
        curand_init(seed + 2000000, 100 * i, 0, &stateYForSample);

        float yCenter = 0.5f * (device_ymax - device_ymin) + device_ymin;
        float normalizedMaxWidth = 3.0f;
        float x, y;

        while (true) {
            float randomValue = curand_uniform(&stateX);
            x = device_xmin + randomValue * (device_xmax - device_xmin); 

            randomValue = curand_uniform(&stateY);
            float yProposed = 2.0f * (randomValue - 0.5f) * normalizedMaxWidth;
            y = yProposed * sheatThickness + yCenter; 

            float targetProbability  = 1.0f / (coefFadeev * cos(x / sheatThickness) + sqrt(1.0f + pow(coefFadeev, 2)) * cosh(yProposed));
            float targetMax          = 1.0f / (coefFadeev * cos(x / sheatThickness) + sqrt(1.0f + pow(coefFadeev, 2))) + device_EPS;
            float sampledProbability = curand_uniform(&stateYForSample);
            if (targetProbability / targetMax > sampledProbability) break;
        }
        
        particle[i + nStart].x = x; 
        particle[i + nStart].y = y;
    }
}

void InitializeParticle::fadeevForPosition(
    unsigned long long nStart, 
    unsigned long long nEnd, 
    int seed, 
    float sheatThickness, float coefFadeev, 
    thrust::device_vector<Particle>& particlesSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nEnd - nStart + threadsPerBlock.x - 1) / threadsPerBlock.x);

    fadeevForPosition_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        sheatThickness, coefFadeev, 
        nStart, nEnd, seed
    );

    cudaDeviceSynchronize();
}

