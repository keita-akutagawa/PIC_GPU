#include "initialize_particle.hpp"
#include <thrust/transform.h>
#include <thrust/random.h>
#include <curand_kernel.h>
#include <cmath>
#include <random>


__global__ void uniformForPositionX_kernel(
    Particle* particle, 
    const int nStart, const int nEnd, const int seed
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, i, 0, &state);
        double x = curand_uniform(&state) * (device_xmax - device_xmin) + device_xmin;
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
    const double bulkVxSpecies, const double bulkVySpecies, const double bulkVzSpecies, const double vThSpecies, 
    const int nStart, const int nEnd, const int seed
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, 1000 * i, 0, &state);

        double vx, vy, vz;

        while (true) {
            vx = bulkVxSpecies + curand_normal(&state) * vThSpecies;
            vy = bulkVySpecies + curand_normal(&state) * vThSpecies;
            vz = bulkVzSpecies + curand_normal(&state) * vThSpecies;

            if (vx * vx + vy * vy + vz * vz < device_c * device_c) break;
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

//////////

void InitializeParticle::uniformForPositionX_cpu(
    int nStart, 
    int nEnd, 
    int seed, 
    thrust::host_vector<Particle>& host_particlesSpecies,
    thrust::device_vector<Particle>& particlesSpecies
)
{
    std::mt19937_64 mt64(seed);
    std::uniform_real_distribution<double> set_x(1e-20, 1.0 - 1e-20);
    for (int i = nStart; i < nEnd; i++) {
        double x = set_x(mt64) * (xmax - xmin);
        host_particlesSpecies[i].x = x;
    }

    particlesSpecies = host_particlesSpecies;
}


void maxwellDistributionForVelocity_cpu(
        double bulkVxSpecies, 
        double bulkVySpecies, 
        double bulkVzSpecies, 
        double vThSpecies, 
        int nStart, 
        int nEnd, 
        int seed, 
        thrust::host_vector<Particle>& host_particlesSpecies,
        thrust::device_vector<Particle>& particlesSpecies
)
{
    std::mt19937_64 mt64Vx(seed);
    std::normal_distribution<double> set_vx(bulkVxSpecies, vThSpecies);
    std::mt19937_64 mt64Vy(seed + 10000);
    std::normal_distribution<double> set_vy(bulkVySpecies, vThSpecies);
    std::mt19937_64 mt64Vz(seed + 100000);
    std::normal_distribution<double> set_vz(bulkVzSpecies, vThSpecies);

    for (int i = nStart; i < nEnd; i++) {
        double vx;
        double vy;
        double vz;

        while (true) {
            vx = set_vx(mt64Vx);
            vy = set_vy(mt64Vy);
            vz = set_vz(mt64Vz);

            if (vx * vx + vy * vy + vz * vz < c * c) break;
        }

        host_particlesSpecies[i].vx = vx;
        host_particlesSpecies[i].vy = vy;
        host_particlesSpecies[i].vz = vz;
        host_particlesSpecies[i].gamma = sqrt(1.0 + (vx * vx + vy * vy + vz * vz) / (c * c));
    }

    particlesSpecies = host_particlesSpecies;
}

