#include <vector>
#include "particle_struct.hpp"
#include "const.hpp"


class ParticlePush
{
private: 

public:

    void pushPosition(
        std::vector<Particle>& particlesIon, 
        std::vector<Particle>& particlesElectron, 
        double dt
    );

    void pushVelocityForPeriodicBoundary(
        std::vector<Particle>& particlesIon, 
        std::vector<Particle>& particlesElectron, 
        const std::vector<std::vector<std::vector<double>>>& B, 
        const std::vector<std::vector<std::vector<double>>>& E, 
        double dt
    );

    void pushVelocityForWallBoundary(
        std::vector<Particle>& particlesIon, 
        std::vector<Particle>& particlesElectron, 
        const std::vector<std::vector<std::vector<double>>>& B, 
        const std::vector<std::vector<std::vector<double>>>& E, 
        double dt
    );

private:

    void pushPositionOfOneSpecies(
        std::vector<Particle>& particlesSpecies, 
        int totalNumSpecies, 
        double dt
    );

    void pushVelocityOfOneSpeciesForPeriodicBoundary(
        std::vector<Particle>& particlesSpecies, 
        const std::vector<std::vector<std::vector<double>>>& B, 
        const std::vector<std::vector<std::vector<double>>>& E, 
        double q, double m, int totalNumSpecies, 
        double dt
    );

    ParticleField getParticleFieldsForPeriodicBoundary(
        const std::vector<std::vector<std::vector<double>>>& B, 
        const std::vector<std::vector<std::vector<double>>>& E, 
        const Particle& particle
    );

    void pushVelocityOfOneSpeciesForWallBoundary(
        std::vector<Particle>& particlesSpecies, 
        const std::vector<std::vector<std::vector<double>>>& B, 
        const std::vector<std::vector<std::vector<double>>>& E, 
        double q, double m, int totalNumSpecies, 
        double dt
    );

    ParticleField getParticleFieldsForWallBoundary(
        const std::vector<std::vector<std::vector<double>>>& B, 
        const std::vector<std::vector<std::vector<double>>>& E, 
        const Particle& particle
    );
};


