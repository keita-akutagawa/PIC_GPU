#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "const.hpp"
#include "initialize_particle.hpp"
#include "particle_push.hpp"
#include "field_solver.hpp"
#include "current_calculator.hpp"
#include "boundary.hpp"
#include "moment_calculater.hpp"
#include "filter.hpp"
#include "particle_struct.hpp"
#include "field_parameter_struct.hpp"
#include "moment_struct.hpp"
#include "mpi.hpp"


class PIC2D
{
private:
    MPIInfo& mPIInfo;
    MPIInfo* device_mPIInfo;

    thrust::device_vector<Particle> particlesIon;
    thrust::device_vector<Particle> particlesElectron;
    thrust::device_vector<ElectricField> E;
    thrust::device_vector<ElectricField> tmpE;
    thrust::device_vector<MagneticField> B;
    thrust::device_vector<MagneticField> tmpB;
    thrust::device_vector<CurrentField> current;
    thrust::device_vector<CurrentField> tmpCurrent;
    thrust::device_vector<ZerothMoment> zerothMomentIon;
    thrust::device_vector<ZerothMoment> zerothMomentElectron;
    thrust::device_vector<FirstMoment> firstMomentIon;
    thrust::device_vector<FirstMoment> firstMomentElectron;
    thrust::device_vector<SecondMoment> secondMomentIon;
    thrust::device_vector<SecondMoment> secondMomentElectron;

    thrust::host_vector<Particle> host_particlesIon;
    thrust::host_vector<Particle> host_particlesElectron;
    thrust::host_vector<ElectricField> host_E;
    thrust::host_vector<MagneticField> host_B; 
    thrust::host_vector<CurrentField> host_current;
    thrust::host_vector<ZerothMoment> host_zerothMomentIon;
    thrust::host_vector<ZerothMoment> host_zerothMomentElectron;
    thrust::host_vector<FirstMoment> host_firstMomentIon;
    thrust::host_vector<FirstMoment> host_firstMomentElectron;
    thrust::host_vector<SecondMoment> host_secondMomentIon;
    thrust::host_vector<SecondMoment> host_secondMomentElectron;

    InitializeParticle initializeParticle;
    ParticlePush particlePush;
    FieldSolver fieldSolver;
    CurrentCalculator currentCalculator;
    Boundary boundary;
    MomentCalculater momentCalculater;
    Filter filter;

public:
    PIC2D(MPIInfo& mPIInfo);
    
    virtual void initialize();
    
    void oneStep_periodicXY();

    void saveFields(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

    void saveFullMoments(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

    void saveZerothMoments(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

    void saveFirstMoments(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

    void saveSecondMoments(
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

    void calculateFullMoments();

    void calculateZerothMoments();

    void calculateFirstMoments();

    void calculateSecondMoments();

};


