#include <cmath>
#include "current_calculater.hpp"


void CurrentCalculater::resetCurrent(
    std::vector<std::vector<double>>& current
)
{
    for (int comp = 0; comp < 3; comp++) {
        for (int i = 0; i < nx; i++) {
            current[comp][i] = 0.0;
        }
    }
}


void CurrentCalculater::calculateCurrent(
    std::vector<std::vector<double>>& current, 
    const std::vector<Particle>& particlesIon, 
    const std::vector<Particle>& particlesEleectron
)
{
    calculateCurrentOfOneSpecies(
        current, particlesIon, qIon, totalNumIon
    );
    calculateCurrentOfOneSpecies(
        current, particlesEleectron, qElectron, totalNumElectron
    );
}


void CurrentCalculater::calculateCurrentOfOneSpecies(
    std::vector<std::vector<double>>& current, 
    const std::vector<Particle>& particlesSpecies,  
    double q, double totalNumSpecies
)
{
    double cx1, cx2, xIndex1, xIndex2;
    double xOverDx;
    double qOverGamma, qVxOverGamma, qVyOverGamma, qVzOverGamma;

    for (int i = 0; i < totalNumSpecies; i++) {
        xOverDx = particlesSpecies[i].x / dx;

        xIndex1 = std::floor(xOverDx);
        xIndex2 = xIndex1 + 1;
        xIndex2 = (xIndex2 == nx) ? 0 : xIndex2;

        cx1 = xOverDx - xIndex1;
        cx2 = 1.0 - cx1;

        qOverGamma = q / particlesSpecies[i].gamma;
        qVxOverGamma = qOverGamma * particlesSpecies[i].vx;
        qVyOverGamma = qOverGamma * particlesSpecies[i].vy;
        qVzOverGamma = qOverGamma * particlesSpecies[i].vz;

        current[0][xIndex1] += qVxOverGamma * cx2;
        current[0][xIndex2] += qVxOverGamma * cx1;

        current[1][xIndex1] += qVyOverGamma * cx2;
        current[1][xIndex2] += qVyOverGamma * cx1;

        current[2][xIndex1] += qVzOverGamma * cx2;
        current[2][xIndex2] += qVzOverGamma * cx1;
    }
}


