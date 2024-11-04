#include <thrust/device_vector.h>
#include <cmath>
#include "particle_struct.hpp"
#include "field_parameter_struct.hpp"
#include "const.hpp"
#include "mpi.hpp"


class ParticlePush
{
private: 
    MPIInfo& mPIInfo; 

public:
    ParticlePush(MPIInfo& mPIInfo); 

    void pushVelocity(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron, 
        const thrust::device_vector<MagneticField>& B, 
        const thrust::device_vector<ElectricField>& E, 
        const float dt
    );
    void pushPosition(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron, 
        const float dt
    );

private:

    void pushVelocityOfOneSpecies(
        thrust::device_vector<Particle>& particlesSpecies, 
        const thrust::device_vector<MagneticField>& B,
        const thrust::device_vector<ElectricField>& E, 
        const float q, const float m, const unsigned long long existNumSpecies, 
        const float dt
    );

    void pushPositionOfOneSpecies(
        thrust::device_vector<Particle>& particlesSpecies, 
        const unsigned long long existNumSpecies, 
        unsigned int& numForSendParticlesSpeciesLeft, 
        unsigned int& numForSendParticlesSpeciesRight, 
        unsigned int& numForSendParticlesSpeciesDown, 
        unsigned int& numForSendParticlesSpeciesUp, 
        unsigned int& numForSendParticlesSpeciesCornerLeftDown, 
        unsigned int& numForSendParticlesSpeciesCornerRightDown, 
        unsigned int& numForSendParticlesSpeciesCornerLeftUp, 
        unsigned int& numForSendParticlesSpeciesCornerRightUp, 
        const float dt
    );
};


