#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "const.hpp"
#include "particle_struct.hpp"


class InitializeParticle
{
private:

public:
    void uniformForPositionX(
        unsigned long long nStart, 
        unsigned long long nEnd, 
        int seed, 
        thrust::device_vector<Particle>& particlesSpecies
    );

    void uniformForPositionY(
        unsigned long long nStart, 
        unsigned long long nEnd, 
        int seed, 
        thrust::device_vector<Particle>& particlesSpecies
    );

    void uniformForPositionYDetail(
        unsigned long long nStart, 
        unsigned long long nEnd, 
        int seed, 
        float ymin, 
        float ymax, 
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

    void fadeevForPosition(
        unsigned long long nStart, 
        unsigned long long nEnd, 
        int seed, 
        float sheatThickness, float coefFadeev, 
        thrust::device_vector<Particle>& particlesSpecies
    );

private:

};

