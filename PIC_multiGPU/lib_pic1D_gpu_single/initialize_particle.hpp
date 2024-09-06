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
        double xmin, 
        double xmax, 
        int seed, 
        thrust::device_vector<Particle>& particlesSpecies
    );

    void maxwellDistributionForVelocity(
        double bulkVxSpecies, 
        double bulkVySpecies, 
        double bulkVzSpecies, 
        double vThSpecies, 
        int nStart, 
        int nEnd, 
        int seed, 
        thrust::device_vector<Particle>& particlesSpecies
    );

private:

};

