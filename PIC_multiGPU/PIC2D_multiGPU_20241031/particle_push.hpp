#include <thrust/device_vector.h>
#include <cmath>
#include "particle_struct.hpp"
#include "field_parameter_struct.hpp"
#include "const.hpp"
#include "mpi.hpp"


class ParticlePush
{
private: 
    MPIInfo mPIInfo; 

public:
    ParticlePush(MPIInfo& mPIInfo); 

    void pushVelocity(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron, 
        const thrust::device_vector<MagneticField>& B, 
        const thrust::device_vector<ElectricField>& E, 
        float dt
    );
    void pushPosition(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron, 
        float dt
    );

private:

    void pushVelocityOfOneSpecies(
        thrust::device_vector<Particle>& particlesSpecies, 
        const thrust::device_vector<MagneticField>& B,
        const thrust::device_vector<ElectricField>& E, 
        float q, float m, unsigned long long totalNumSpecies, 
        float dt
    );

    void pushPositionOfOneSpecies(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long totalNumSpecies, 
        float dt
    );
};


