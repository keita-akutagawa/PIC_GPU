#include <vector>
#include "const.hpp"
#include "particle_struct.hpp"


class Filter
{
private:

    std::vector<std::vector<double>> rho;

public:
    Filter() :
        rho(nx, std::vector<double>(ny, 0.0))
        {}

    void langdonMarderCorrection(
        std::vector<std::vector<double>>& F, 
        std::vector<std::vector<std::vector<double>>>& E, 
        const std::vector<Particle>& particlesIon, 
        const std::vector<Particle>& particlesElectron
    );

private:
    void resetRho();

    void calculateRho(
        const std::vector<Particle>& particlesIon, 
        const std::vector<Particle>& particlesElectron
    );

    void calculateRhoOfOneSpecies(
        const std::vector<Particle>& particlesSpecies, 
        double q, int nStart, int nEnd
    );
};


