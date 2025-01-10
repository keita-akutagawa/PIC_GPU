#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "const.hpp"
#include "particle_struct.hpp"
#include "mpi.hpp" 


class InitializeParticle
{
private:
    MPIInfo& mPIInfo;

public:
    InitializeParticle(MPIInfo& mPIInfo);

    void uniformForPosition_x(
        unsigned long long nStart, 
        unsigned long long nEnd, 
        float xmin, 
        float xmax, 
        int seed, 
        thrust::device_vector<Particle>& particlesSpecies
    );

    void uniformForPosition_y(
        unsigned long long nStart, 
        unsigned long long nEnd, 
        float ymin, 
        float ymax, 
        int seed, 
        thrust::device_vector<Particle>& particlesSpecies
    );

    void maxwellDistributionForVelocity(
        float bulkVxSpecies, 
        float bulkVySpecies, 
        float bulkVzSpecies, 
        float vxThSpecies, 
        float vyThSpecies, 
        float vzThSpecies, 
        unsigned long long nStart, 
        unsigned long long nEnd, 
        int seed, 
        thrust::device_vector<Particle>& particlesSpecies
    );

    /*
    void harrisForPositionY(
        unsigned long long nStart, 
        unsigned long long nEnd, 
        int seed, 
        float sheatThickness, 
        thrust::device_vector<Particle>& particlesSpecies
    );

    void harrisBackgroundForPositionY(
        unsigned long long nStart, 
        unsigned long long nEnd, 
        int seed, 
        float sheatThickness, 
        thrust::device_vector<Particle>& particlesSpecies
    );
    */

private:

};

