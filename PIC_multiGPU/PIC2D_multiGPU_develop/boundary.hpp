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

    thrust::device_vector<MagneticField> sendMagneticFieldLeft; 
    thrust::device_vector<MagneticField> sendMagneticFieldRight; 
    thrust::device_vector<MagneticField> recvMagneticFieldLeft; 
    thrust::device_vector<MagneticField> recvMagneticFieldRight;

    thrust::device_vector<MagneticField> sendMagneticFieldDown; 
    thrust::device_vector<MagneticField> sendMagneticFieldUp; 
    thrust::device_vector<MagneticField> recvMagneticFieldDown; 
    thrust::device_vector<MagneticField> recvMagneticFieldUp; 

    thrust::device_vector<ElectricField> sendElectricFieldLeft; 
    thrust::device_vector<ElectricField> sendElectricFieldRight; 
    thrust::device_vector<ElectricField> recvElectricFieldLeft; 
    thrust::device_vector<ElectricField> recvElectricFieldRight; 

    thrust::device_vector<ElectricField> sendElectricFieldDown; 
    thrust::device_vector<ElectricField> sendElectricFieldUp; 
    thrust::device_vector<ElectricField> recvElectricFieldDown; 
    thrust::device_vector<ElectricField> recvElectricFieldUp; 

    thrust::device_vector<CurrentField> sendCurrentFieldLeft; 
    thrust::device_vector<CurrentField> sendCurrentFieldRight; 
    thrust::device_vector<CurrentField> recvCurrentFieldLeft; 
    thrust::device_vector<CurrentField> recvCurrentFieldRight; 

    thrust::device_vector<CurrentField> sendCurrentFieldDown; 
    thrust::device_vector<CurrentField> sendCurrentFieldUp; 
    thrust::device_vector<CurrentField> recvCurrentFieldDown; 
    thrust::device_vector<CurrentField> recvCurrentFieldUp; 

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


