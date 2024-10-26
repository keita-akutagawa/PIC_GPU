#include <thrust/sort.h>
#include "sort_particle.hpp"


void ParticleSorter::sortParticle(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron
)
{
    sortParticleOfOneSpecies(particlesIon);
    sortParticleOfOneSpecies(particlesElectron);
}


void ParticleSorter::sortParticleOfOneSpecies(
    thrust::device_vector<Particle>& particlesSpecies
)
{
    thrust::sort(particlesSpecies.begin(), particlesSpecies.end());
    cudaDeviceSynchronize();
}


