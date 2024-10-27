#include <thrust/device_vector.h>
#include "const.hpp"
#include "particle_struct.hpp"
#include "field_parameter_struct.hpp"
#include "mpi.hpp"


class Boundary
{
private:
    MPIInfo mPIInfo; 

    thrust::device_vector<Particle> sendParticlesSpeciesLeftToRight;
    thrust::device_vector<Particle> sendParticlesSpeciesRightToLeft;
    thrust::device_vector<Particle> recvParticlesSpeciesLeftToRight;
    thrust::device_vector<Particle> recvParticlesSpeciesRightToLeft;

    thrust::device_vector<Particle> sendParticlesSpeciesUpToDown;
    thrust::device_vector<Particle> sendParticlesSpeciesDownToUp;
    thrust::device_vector<Particle> recvParticlesSpeciesUpToDown;
    thrust::device_vector<Particle> recvParticlesSpeciesDownToUp;

    thrust::device_vector<unsigned long long> device_countForSendSpeciesLeftToRight;
    thrust::device_vector<unsigned long long> device_countForSendSpeciesRightToLeft;
    unsigned long long countForSendSpeciesLeftToRight;
    unsigned long long countForSendSpeciesRightToLeft;
    unsigned long long countForRecvSpeciesLeftToRight;
    unsigned long long countForRecvSpeciesRightToLeft;

    thrust::device_vector<unsigned long long> device_countForSendSpeciesUpToDown;
    thrust::device_vector<unsigned long long> device_countForSendSpeciesDownToUp;
    unsigned long long countForSendSpeciesUpToDown;
    unsigned long long countForSendSpeciesDownToUp;
    unsigned long long countForRecvSpeciesUpToDown;
    unsigned long long countForRecvSpeciesDownToUp;

public:
    Boundary(MPIInfo& mPIInfo);

    void boundaryForinitialize(
        thrust::device_vector<Particle>& particlesSpecies,
        unsigned long long& existNumSpecies
    );

    void periodicBoundaryParticleXY(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particesElectron
    );
    void periodicBoundaryParticleOfOneSpeciesX(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies
    );
    void periodicBoundaryParticleOfOneSpeciesY(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies
    );

    void periodicBoundaryBX(
        thrust::device_vector<MagneticField>& B
    );
    void periodicBoundaryBY(
        thrust::device_vector<MagneticField>& B
    );

    void periodicBoundaryEX(
        thrust::device_vector<ElectricField>& E
    );
    void periodicBoundaryEY(
        thrust::device_vector<ElectricField>& E
    );

    void periodicBoundaryCurrentX(
        thrust::device_vector<CurrentField>& current
    );
    void periodicBoundaryCurrentY(
        thrust::device_vector<CurrentField>& current
    );

private:

};


