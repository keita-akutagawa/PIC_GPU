#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "const.hpp"
#include "particle_struct.hpp"


class InitializeParticle
{
private:

public:
    void uniformForPositionX(
        int nStart, 
        int nEnd, 
        int seed, 
        thrust::device_vector<Particle>& particlesSpecies
    );

    void maxwellDistributionForVelocity(
        float bulkVxSpecies, 
        float bulkVySpecies, 
        float bulkVzSpecies, 
        float vThSpecies, 
        int nStart, 
        int nEnd, 
        int seed, 
        thrust::device_vector<Particle>& particlesSpecies
    );

    void uniformForPositionX_cpu(
        int nStart, 
        int nEnd, 
        int seed, 
        thrust::host_vector<Particle>& host_particlesSpecies,
        thrust::device_vector<Particle>& particlesSpecies
    );

    void maxwellDistributionForVelocity_cpu(
        float bulkVxSpecies, 
        float bulkVySpecies, 
        float bulkVzSpecies, 
        float vThSpecies, 
        int nStart, 
        int nEnd, 
        int seed, 
        thrust::host_vector<Particle>& host_particlesSpecies,
        thrust::device_vector<Particle>& particlesSpecies
    );

private:

};

