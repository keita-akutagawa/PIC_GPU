#include <thrust/device_vector.h>
#include "const.hpp"
#include "particle_struct.hpp"
#include "field_parameter_struct.hpp"


class Boundary
{
private:

public:

    void periodicBoundaryParticleX(
        thrust::device_vector<Particle>& particlesIon,
        thrust::device_vector<Particle>& particlesElectron
    );
    void conductingWallBoundaryParticleX(
        thrust::device_vector<Particle>& particlesIon,
        thrust::device_vector<Particle>& particlesElectron
    );

    void periodicBoundaryBX(
        thrust::device_vector<MagneticField>& magneticField
    );
    void conductingWallBoundaryBX();

    void periodicBoundaryEX(
        thrust::device_vector<ElectricField>& electricField
    );
    void conductingWallBoundaryEX();

    void periodicBoundaryCurrentX(
        thrust::device_vector<CurrentField>& currentField
    );
    void conductingWallBoundaryCurrentX();

private:

};


