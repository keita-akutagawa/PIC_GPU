#include <vector>
#include "particle_struct.hpp"
#include "const.hpp"


class ParticlePush
{
private: 

public:

    void pushVelocity(
        std::vector<Particle>& particlesIon, 
        std::vector<Particle>& particlesElectron, 
        const std::vector<std::vector<double>>& B, 
        const std::vector<std::vector<double>>& E, 
        double dt
    );
    void pushPosition(
        std::vector<Particle>& particlesIon, 
        std::vector<Particle>& particlesElectron, 
        double dt
    );

private:

    void pushVelocityOfOneSpecies(
        std::vector<Particle>& particlesSpecies, 
        const std::vector<std::vector<double>>& B, 
        const std::vector<std::vector<double>>& E, 
        double q, double m, int totalNumSpecies, 
        double dt
    );

    ParticleField getParticleFields(
        const std::vector<std::vector<double>>& B, 
        const std::vector<std::vector<double>>& E, 
        const Particle& particle
    );

    void pushPositionOfOneSpecies(
        std::vector<Particle>& particlesSpecies, 
        int totalNumSpecies, 
        double dt
    );
};


