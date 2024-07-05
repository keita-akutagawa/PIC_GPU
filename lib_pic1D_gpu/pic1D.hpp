#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include <string>
#include "const.hpp"
#include "initialize_particle.hpp"
#include "particle_push.hpp"
#include "field_solver.hpp"
#include "current_calculater.hpp"
#include "boundary.hpp"
#include "particle_struct.hpp"
#include "field_parameter_struct.hpp"
#include "moment_struct.hpp"


class PIC1D
{
private:
    thrust::device_vector<Particle> particlesIon;
    thrust::device_vector<Particle> particlesElectron;
    thrust::device_vector<ElectricField> E;
    thrust::device_vector<MagneticField> B;
    thrust::device_vector<CurrentField> current;
    thrust::device_vector<ElectricField> tmpE;
    thrust::device_vector<MagneticField> tmpB;
    thrust::device_vector<CurrentField> tmpCurrent;
    thrust::device_vector<ZerothMoment> zerothMomentIon;
    thrust::device_vector<ZerothMoment> zerothMomentElectron;
    thrust::device_vector<FirstMoment> firstMomentIon;
    thrust::device_vector<FirstMoment> firstMomentElectron;
    thrust::device_vector<SecondMoment> secondMomentIon;
    thrust::device_vector<SecondMoment> secondMomentElectron;

    InitializeParticle initializeParticle;
    ParticlePush particlePush;
    FieldSolver fieldSolver;
    CurrentCalculater currentCalculater;
    Boundary boundary;

    thrust::host_vector<Particle> host_particleIon;
    thrust::host_vector<Particle> host_particleElectron;
    thrust::host_vector<ElectricField> host_E;
    thrust::host_vector<MagneticField> host_B; 
    thrust::host_vector<CurrentField> host_current;
    thrust::host_vector<ZerothMoment> host_zerothMomentIon;
    thrust::host_vector<ZerothMoment> host_zerothMomentElectron;
    thrust::host_vector<FirstMoment> host_firstMomentIon;
    thrust::host_vector<FirstMoment> host_firstMomentElectron;
    thrust::host_vector<SecondMoment> host_secondMomentIon;
    thrust::host_vector<SecondMoment> host_secondMomentElectron;


public:
    PIC1D();
    
    virtual void initialize();
    
    void oneStep();

    void saveFields(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

    void saveParticle(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

private:

};


