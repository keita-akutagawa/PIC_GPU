#include <thrust/device_vector.h>
#include "particle_struct.hpp"
#include "field_parameter_struct.hpp"
#include "const.hpp"


class ParticlePush
{
private: 

public:

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


