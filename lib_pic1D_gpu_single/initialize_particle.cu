#include "initialize_particle.hpp"
#include <thrust/transform.h>
#include <thrust/random.h>
#include <curand_kernel.h>
#include <cmath>


__global__ void uniformForPositionX_kernel(
    Particle* particle, 
    const int nStart, const int nEnd, const int seed
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, i, 0, &state);
        float x = curand_uniform(&state) * (device_xmax - device_xmin) + device_xmin;
        particle[i + nStart].x = x;
    }
}


void InitializeParticle::uniformForPositionX(
    int nStart, 
    int nEnd, 
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



__global__ void maxwellDistributionForVelocity_kernel(
    Particle* particle, 
    const float bulkVxSpecies, const float bulkVySpecies, const float bulkVzSpecies, const float vThSpecies, 
    const int nStart, const int nEnd, const int seed
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, 1000 * i, 0, &state);

        float vx, vy, vz;

        while (true) {
            vx = bulkVxSpecies + curand_normal(&state) * vThSpecies;
            vy = bulkVySpecies + curand_normal(&state) * vThSpecies;
            vz = bulkVzSpecies + curand_normal(&state) * vThSpecies;

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
    float vThSpecies, 
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

//////////

void InitializeParticle::uniformForPositionX_cpu(
    int nStart, 
    int nEnd, 
    int seed, 
    thrust::device_vector<Particle>& particlesSpecies
)
{

}


void maxwellDistributionForVelocity_cpu(
        float bulkVxSpecies, 
        float bulkVySpecies, 
        float bulkVzSpecies, 
        float vThSpecies, 
        int nStart, 
        int nEnd, 
        int seed, 
        thrust::device_vector<Particle>& particlesSpecies
)
{
    
}

