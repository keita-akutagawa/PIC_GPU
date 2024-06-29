#include <vector>
#include "const.hpp"
#include "particle_struct.hpp"


class CurrentCalculater
{
private: 

public: 
    void resetCurrent(
        std::vector<std::vector<double>>& current
    );

    void calculateCurrent(
        std::vector<std::vector<double>>& current, 
        const std::vector<Particle>& particlesIon, 
        const std::vector<Particle>& particlesEleectron
    );

private:

    void calculateCurrentOfOneSpecies(
        std::vector<std::vector<double>>& current, 
        const std::vector<Particle>& particlesSpecies, 
        double q, double totalNumSpecies
    );
};

