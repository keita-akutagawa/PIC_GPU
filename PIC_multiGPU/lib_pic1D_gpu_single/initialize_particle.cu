#include "initialize_particle.hpp"
#include <thrust/transform.h>
#include <thrust/random.h>
#include <curand_kernel.h>
#include <cmath>
#include <random>


__global__ void uniformForPositionX_kernel(
    Particle* particle, 
    const int nStart, const int nEnd, 
    const double xmin, const double xmax, 
    const int seed
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, i, 0, &state);
        double x = curand_uniform(&state) * (xmax - xmin) + xmin;
        particle[i + nStart].x = x;
        particle[i + nStart].isExist = true;
    }
}


void InitializeParticle::uniformForPositionX(
    int nStart, 
    int nEnd, 
    double xmin, 
    double xmax, 
    int seed, 
    thrust::device_vector<Particle>& particlesSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nEnd - nStart + threadsPerBlock.x - 1) / threadsPerBlock.x);

    uniformForPositionX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        nStart, nEnd, 
        xmin, xmax, 
        seed
    );

    cudaDeviceSynchronize();
}



__global__ void maxwellDistributionForVelocity_kernel(
    Particle* particle, 
    const double bulkVxSpecies, const double bulkVySpecies, const double bulkVzSpecies, const double vThSpecies, 
    const int nStart, const int nEnd, const int seed
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

        double vx, vy, vz;

        while (true) {
            vx = bulkVxSpecies + curand_normal(&stateVx) * vThSpecies;
            vy = bulkVySpecies + curand_normal(&stateVy) * vThSpecies;
            vz = bulkVzSpecies + curand_normal(&stateVz) * vThSpecies;

            if (vx * vx + vy * vy + vz * vz < device_c * device_c) break;
        }

        particle[i + nStart].vx = vx;
        particle[i + nStart].vy = vy;
        particle[i + nStart].vz = vz;
        particle[i + nStart].gamma = sqrt(1.0f + (vx * vx + vy * vy + vz * vz) / (device_c * device_c));
        particle[i + nStart].isExist = true;
    }
}


void InitializeParticle::maxwellDistributionForVelocity(
    double bulkVxSpecies, 
    double bulkVySpecies, 
    double bulkVzSpecies, 
    double vThSpecies, 
    int nStart, 
    int nEnd, 
    int seed, 
    thrust::device_vector<Particle>& particlesSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nEnd - nStart + threadsPerBlock.x - 1) / threadsPerBlock.x);

    maxwellDistributionForVelocity_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        bulkVxSpecies, bulkVySpecies, bulkVzSpecies, vThSpecies, 
        nStart, nEnd, seed
    );

    cudaDeviceSynchronize();
}

