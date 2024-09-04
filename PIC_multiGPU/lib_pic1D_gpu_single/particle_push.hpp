#include <thrust/device_vector.h>
#include "particle_struct.hpp"
#include "field_parameter_struct.hpp"
#include "const.hpp"
#include "mpi.hpp"


class ParticlePush
{
private: 

public:

    void pushVelocity(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron, 
        const thrust::device_vector<MagneticField>& B, 
        const thrust::device_vector<ElectricField>& E, 
        float dt, MPIInfo& mPIInfo
    );
    void pushPosition(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron, 
        float dt, MPIInfo& mPIInfo
    );

private:

    void pushVelocityOfOneSpecies(
        thrust::device_vector<Particle>& particlesSpecies, 
        const thrust::device_vector<MagneticField>& B,
        const thrust::device_vector<ElectricField>& E, 
        float q, float m, int existNumSpecies, 
        float dt, 
        const float xminForProcs, const float xmaxForProcs
    );

    void pushPositionOfOneSpecies(
        thrust::device_vector<Particle>& particlesSpecies, 
        int existNumSpecies, 
        float dt
    );
};


