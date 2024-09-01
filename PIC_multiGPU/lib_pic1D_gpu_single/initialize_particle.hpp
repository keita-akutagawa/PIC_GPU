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
        float xmin, 
        float xmax, 
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

private:

};

