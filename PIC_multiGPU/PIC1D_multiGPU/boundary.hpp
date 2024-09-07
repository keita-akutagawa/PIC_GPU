#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "const.hpp"
#include "particle_struct.hpp"
#include "field_parameter_struct.hpp"
#include "mpi.hpp"


class Boundary
{
private:
    thrust::device_vector<Particle> sendParticlesIonLeftToRight;
    thrust::device_vector<Particle> sendParticlesIonRightToLeft;
    thrust::device_vector<Particle> sendParticlesElectronLeftToRight; 
    thrust::device_vector<Particle> sendParticlesElectronRightToLeft;

    thrust::host_vector<Particle> host_sendParticlesIonLeftToRight;
    thrust::host_vector<Particle> host_sendParticlesIonRightToLeft;
    thrust::host_vector<Particle> host_sendParticlesElectronLeftToRight; 
    thrust::host_vector<Particle> host_sendParticlesElectronRightToLeft; 

    thrust::host_vector<Particle> host_recvParticlesIonLeftToRight;
    thrust::host_vector<Particle> host_recvParticlesIonRightToLeft;
    thrust::host_vector<Particle> host_recvParticlesElectronLeftToRight; 
    thrust::host_vector<Particle> host_recvParticlesElectronRightToLeft; 

public:
    Boundary();

    void periodicBoundaryParticleX(
        thrust::device_vector<Particle>& particlesIon,
        thrust::device_vector<Particle>& particlesElectron, 
        MPIInfo& mPIInfo
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


