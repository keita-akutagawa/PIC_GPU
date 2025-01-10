#include <thrust/device_vector.h>
#include <thrust/partition.h>
#include <thrust/transform_reduce.h>
#include "const.hpp"
#include "particle_struct.hpp"
#include "field_parameter_struct.hpp"
#include "mpi.hpp"


class Boundary
{
private:
    MPIInfo& mPIInfo; 

    thrust::device_vector<Particle> sendParticlesSpeciesLeft; 
    thrust::device_vector<Particle> sendParticlesSpeciesRight; 
    thrust::device_vector<Particle> sendParticlesSpeciesDown; 
    thrust::device_vector<Particle> sendParticlesSpeciesUp; 
    thrust::device_vector<Particle> recvParticlesSpeciesLeft; 
    thrust::device_vector<Particle> recvParticlesSpeciesRight; 
    thrust::device_vector<Particle> recvParticlesSpeciesDown; 
    thrust::device_vector<Particle> recvParticlesSpeciesUp; 

public:
    Boundary(MPIInfo& mPIInfo);

    void boundaryForInitializeParticle_xy(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );
    void boundaryForInitializeParticleOfOneSpecies_x(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies, 
        unsigned int& numForSendSpeciesLeftToRight, 
        unsigned int& numForSendSpeciesRightToLeft, 
        unsigned int& numForRecvSpeciesLeftToRight, 
        unsigned int& numForRecvSpeciesRightToLeft
    );
    void boundaryForInitializeParticleOfOneSpecies_y(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies, 
        unsigned int& numForSendSpeciesDownToUp, 
        unsigned int& numForSendSpeciesUpToDown, 
        unsigned int& numForRecvSpeciesDownToUp, 
        unsigned int& numForRecvSpeciesUpToDown
    );

    void periodicBoundaryParticle_xy(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );
    void periodicBoundaryParticleOfOneSpecies_x(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies, 
        unsigned int& numForSendSpeciesLeft, 
        unsigned int& numForSendSpeciesRight, 
        unsigned int& numForRecvSpeciesLeft, 
        unsigned int& numForRecvSpeciesRight
    );
    void periodicBoundaryParticleOfOneSpecies_y(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies, 
        unsigned int& numForSendSpeciesDown, 
        unsigned int& numForSendSpeciesUp, 
        unsigned int& numForRecvSpeciesDown, 
        unsigned int& numForRecvSpeciesUp
    );

    void modifySendNumParticlesSpecies(
        const unsigned int& numForSendParticlesSpeciesCornerLeftDown, 
        const unsigned int& numForSendParticlesSpeciesCornerRightDown, 
        const unsigned int& numForSendParticlesSpeciesCornerLeftUp, 
        const unsigned int& numForSendParticlesSpeciesCornerRightUp, 
        unsigned int& numForRecvParticlesSpeciesCornerLeftDown, 
        unsigned int& numForRecvParticlesSpeciesCornerRightDown, 
        unsigned int& numForRecvParticlesSpeciesCornerLeftUp, 
        unsigned int& numForRecvParticlesSpeciesCornerRightUp, 
        unsigned int& numForSendParticlesSpeciesDown, 
        unsigned int& numForSendParticlesSpeciesUp
    );
    

    void periodicBoundaryB_x(
        thrust::device_vector<MagneticField>& B
    );
    void periodicBoundaryB_y(
        thrust::device_vector<MagneticField>& B
    );

    void periodicBoundaryE_x(
        thrust::device_vector<ElectricField>& E
    );
    void periodicBoundaryE_y(
        thrust::device_vector<ElectricField>& E
    );

    void periodicBoundaryCurrent_x(
        thrust::device_vector<CurrentField>& current
    );
    void periodicBoundaryCurrent_y(
        thrust::device_vector<CurrentField>& current
    );

private:

};


