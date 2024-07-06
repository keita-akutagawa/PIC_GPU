#include "initialize_particle.hpp"
#include <thrust/transform.h>
#include <thrust/random.h>
#include <cmath>


__global__ void uniformForPositionX_kernel(
    Particle* particle, 
    const int nStart, const int nEnd, const int seed
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        thrust::default_random_engine rng(i + nStart);
        thrust::uniform_real_distribution<float> dist(device_xmin + 1e-20f, device_xmax - 1e-20f);
        particle[i + nStart].x = dist(rng);
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
        thrust::default_random_engine rng(nStart + i);
        thrust::random::normal_distribution<float> dist_vx(bulkVxSpecies, vThSpecies);
        thrust::random::normal_distribution<float> dist_vy(bulkVySpecies, vThSpecies);
        thrust::random::normal_distribution<float> dist_vz(bulkVzSpecies, vThSpecies);

        float vx, vy, vz;

        while (true) {
            vx = dist_vx(rng);
            vy = dist_vy(rng);
            vz = dist_vz(rng);

            if (vx * vx + vy * vy + vz * vz < device_c * device_c) break;
        }

        particle[i + nStart].vx = vx;
        particle[i + nStart].vy = vy;
        particle[i + nStart].vz = vz;
        particle[i + nStart].gamma = sqrt(1.0 + (vx * vx + vy * vy + vz * vz) / (device_c * device_c));
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

