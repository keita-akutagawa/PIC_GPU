#include <vector>
#include <string>
#include "const.hpp"
#include "initialize_particle.hpp"
#include "particle_push.hpp"
#include "field_solver.hpp"
#include "current_calculater.hpp"
#include "filter.hpp"
#include "boundary.hpp"


class PIC2DSymXConY
{
private:
    std::vector<Particle> particlesIon;
    std::vector<Particle> particlesElectron;
    std::vector<std::vector<std::vector<double>>> E;
    std::vector<std::vector<std::vector<double>>> B;
    std::vector<std::vector<std::vector<double>>> current;
    std::vector<std::vector<std::vector<double>>> tmpE;
    std::vector<std::vector<std::vector<double>>> tmpB;
    std::vector<std::vector<std::vector<double>>> tmpCurrent;
    std::vector<std::vector<double>> F;
    std::vector<std::vector<double>> zerothMomentIon;
    std::vector<std::vector<double>> zerothMomentElectron;
    std::vector<std::vector<std::vector<double>>> firstMomentIon;
    std::vector<std::vector<std::vector<double>>> firstMomentElectron;
    std::vector<std::vector<std::vector<double>>> secondMomentIon;
    std::vector<std::vector<std::vector<double>>> secondMomentElectron;

    InitializeParticle initializeParticle;
    ParticlePush particlePush;
    FieldSolver fieldSolver;
    CurrentCalculater currentCalculater;
    Filter filter;
    Boundary boundary;

public:
    PIC2DSymXConY() :
        particlesIon(totalNumIon), 
        particlesElectron(totalNumElectron), 
        E(3, std::vector<std::vector<double>>(nx, std::vector<double>(ny, 0.0))), 
        B(3, std::vector<std::vector<double>>(nx, std::vector<double>(ny, 0.0))), 
        current(3, std::vector<std::vector<double>>(nx, std::vector<double>(ny, 0.0))), 
        tmpE(3, std::vector<std::vector<double>>(nx, std::vector<double>(ny, 0.0))), 
        tmpB(3, std::vector<std::vector<double>>(nx, std::vector<double>(ny, 0.0))), 
        tmpCurrent(3, std::vector<std::vector<double>>(nx, std::vector<double>(ny, 0.0))),
        F(nx, std::vector<double>(ny, 0.0)),
        zerothMomentIon(nx, std::vector<double>(ny, 0.0)),
        zerothMomentElectron(nx, std::vector<double>(ny, 0.0)), 
        firstMomentIon(3, std::vector<std::vector<double>>(nx, std::vector<double>(ny, 0.0))),
        firstMomentElectron(3, std::vector<std::vector<double>>(nx, std::vector<double>(ny, 0.0))),  
        secondMomentIon(9, std::vector<std::vector<double>>(nx, std::vector<double>(ny, 0.0))), 
        secondMomentElectron(9, std::vector<std::vector<double>>(nx, std::vector<double>(ny, 0.0)))
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

    void saveEnergy(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

    void saveMoments(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

private:

    void resetMoments();

    void calculateMomentOfOneSpecies(
        std::vector<std::vector<double>>& zerothMomentSpecies, 
        std::vector<std::vector<std::vector<double>>>& firstMomentSpecies, 
        std::vector<std::vector<std::vector<double>>>& secondMomentSpecies, 
        const std::vector<Particle>& particlesSpecies, 
        int totalNumSpecies
    );
};


