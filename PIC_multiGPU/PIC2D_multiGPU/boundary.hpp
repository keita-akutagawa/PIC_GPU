#include <thrust/device_vector.h>
#include "const.hpp"
#include "particle_struct.hpp"
#include "field_parameter_struct.hpp"
#include "mpi.hpp"


class Boundary
{
private:
    MPIInfo& mPIInfo; 

public:
    Boundary(MPIInfo& mPIInfo);

    void boundaryForInitializeParticle(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );
    void boundaryForInitializeParticleOfOneSpeciesX(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies, 
        unsigned int& numForSendSpeciesLeftToRightward, 
        unsigned int& numForSendSpeciesRightToLeftward, 
        unsigned int& numForRecvSpeciesLeftToRightward, 
        unsigned int& numForRecvSpeciesRightToLeftward
    );
    void boundaryForInitializeParticleOfOneSpeciesY(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies, 
        unsigned int& numForSendSpeciesDownToUpward, 
        unsigned int& numForSendSpeciesUpToDownward, 
        unsigned int& numForRecvSpeciesDownToUpward, 
        unsigned int& numForRecvSpeciesUpToDownward
    );

    void periodicBoundaryParticleXY(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );
    void periodicBoundaryParticleOfOneSpeciesX(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies, 
        unsigned int& numForSendSpeciesLeftward, 
        unsigned int& numForSendSpeciesRightward, 
        unsigned int& numForRecvSpeciesLeftward, 
        unsigned int& numForRecvSpeciesRightward
    );
    void periodicBoundaryParticleOfOneSpeciesY(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies, 
        unsigned int& numForSendSpeciesDownward, 
        unsigned int& numForSendSpeciesUpward, 
        unsigned int& numForRecvSpeciesDownward, 
        unsigned int& numForRecvSpeciesUpward
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


