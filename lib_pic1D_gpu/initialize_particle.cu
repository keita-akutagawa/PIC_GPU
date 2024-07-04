#include "initialize_particle.hpp"
#include <thrust/transform.h>
#include <thrust/random.h>
#include <cmath>


struct UniformForPositionXFuctor{
    float xmin; 
    float xmax; 
    int seed;

    __device__ 
    UniformForPositionXFuctor(float xmin, float xmax, int seed)
        : xmin(xmin), xmax(xmax), seed(seed) {}
    
    __device__ 
    float operator()(const int& i) const {
        thrust::default_random_engine rng(seed + i);
        thrust::uniform_real_distribution<float> dist(1e-20, 1.0 - 1e-20);
        return dist(rng) * (xmax - xmin);
    }
};

void InitializeParticle::uniformForPositionX(
    int nStart, 
    int nEnd, 
    int seed, 
    thrust::device_vector<Particle>& particlesSpecies
)
{
    UniformForPositionXFuctor uniformForPositionXFunctor(xmin, xmax, seed);

    thrust::transform(
        thrust::make_counting_iterator(nStart),
        thrust::make_counting_iterator(nEnd),
        particlesSpecies.begin(), 
        [=] __device__ (int i) {
            Particle p;
            p.x = uniformForPositionXFunctor(i);
            return p;
        }
    );
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

        particlesSpecies[i].vx = vx;
        particlesSpecies[i].vy = vy;
        particlesSpecies[i].vz = vz;
        particlesSpecies[i].gamma = sqrt(1.0 + (vx * vx + vy * vy + vz * vz) / (c * c));
    }
}

