#include <thrust/sort.h>
#include "sort_particle.hpp"


void ParticleSorter::sortParticle(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron, 
    thrust::device_vector<Particle>& particlesHeavyIon
)
{
    sortParticleOfOneSpecies(particlesIon);
    sortParticleOfOneSpecies(particlesElectron);
    sortParticleOfOneSpecies(particlesHeavyIon);
}


void ParticleSorter::sortParticleOfOneSpecies(
    thrust::device_vector<Particle>& particlesSpecies
)
{
    thrust::sort(particlesSpecies.begin(), particlesSpecies.end());
    cudaDeviceSynchronize();
}


