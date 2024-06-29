#include <vector>
#include <string>
#include "const.hpp"
#include "initialize_particle.hpp"
#include "particle_push.hpp"
#include "field_solver.hpp"
#include "current_calculater.hpp"
#include "boundary.hpp"


class PIC1D
{
private:
    std::vector<Particle> particlesIon;
    std::vector<Particle> particlesElectron;
    std::vector<std::vector<double>> E;
    std::vector<std::vector<double>> B;
    std::vector<std::vector<double>> current;
    std::vector<std::vector<double>> tmpE;
    std::vector<std::vector<double>> tmpB;
    std::vector<std::vector<double>> tmpCurrent;

    InitializeParticle initializeParticle;
    ParticlePush particlePush;
    FieldSolver fieldSolver;
    CurrentCalculater currentCalculater;
    Boundary boundary;

public:
    PIC1D() :
        particlesIon(totalNumIon), 
        particlesElectron(totalNumElectron), 
        E(3, std::vector<double>(nx, 0.0)), 
        B(3, std::vector<double>(nx, 0.0)), 
        current(3, std::vector<double>(nx, 0.0)), 
        tmpE(3, std::vector<double>(nx, 0.0)), 
        tmpB(3, std::vector<double>(nx, 0.0)), 
        tmpCurrent(3, std::vector<double>(nx, 0.0))
        {}
    
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


