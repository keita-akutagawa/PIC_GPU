#include "initialize_particle.hpp"
#include <thrust/transform.h>
#include <thrust/random.h>
#include <cmath>


__global__ void uniformForPositionX_kernel(
    Particle* particle, 
    const float xmin, const float xmax, 
    const int nStart, const int nEnd, const int seed
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + nStart;

    if (i < nEnd - nStart) {
        thrust::default_random_engine rng(i + seed);
        thrust::uniform_real_distribution<float> dist(1e-20, 1.0 - 1e-20);
        particle[i + nStart].x =  dist(rng) * (xmax - xmin);
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
        xmin, xmax, nStart, nEnd, seed
    );

    cudaDeviceSynchronize();
}



__global__ void maxwellDistributionForVelocity_kernel(
    Particle* particle, 
    const float bulkVxSpecies, const float bulkVySpecies, const float bulkVzSpecies, const float vThSpecies, 
    const int nStart, const int nEnd, const int seed
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + nStart;

    if (i < nEnd - nStart) {
        thrust::default_random_engine rng(seed + i);
        thrust::random::normal_distribution<double> dist_vx(bulkVxSpecies, vThSpecies);
        thrust::random::normal_distribution<double> dist_vy(bulkVySpecies, vThSpecies);
        thrust::random::normal_distribution<double> dist_vz(bulkVzSpecies, vThSpecies);

        double vx, vy, vz;

        while (true) {
            vx = dist_vx(rng);
            vy = dist_vy(rng);
            vz = dist_vz(rng);

            if (vx * vx + vy * vy + vz * vz < c * c) break;
        }

        particle[i + nStart].vx = vx;
        particle[i + nStart].vy = vy;
        particle[i + nStart].vz = vz;
        particle[i + nStart].gamma = sqrt(1.0 + (vx * vx + vy * vy + vz * vz) / (device_c * device_c));
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

