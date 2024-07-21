#include <thrust/device_vector.h>
#include "particle_struct.hpp"
#include "const.hpp"


class ParticleSorter
{
private: 

public:
    void sortParticle(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron, 
        thrust::device_vector<Particle>& particlesHeavyIon 
    );

private:
    void sortParticleOfOneSpecies(
        thrust::device_vector<Particle>& particlesSpecies
    );
};

