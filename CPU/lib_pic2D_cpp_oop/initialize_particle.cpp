#include "initialize_particle.hpp"
#include <random>
#include <cmath>


void InitializeParticle::uniformForPositionX(
    int nStart, 
    int nEnd, 
    int seed, 
    std::vector<Particle>& particlesSpecies
)
{
    std::mt19937_64 mt64(seed);
    std::uniform_real_distribution<double> set_x(1e-20, 1.0 - 1e-20);

    for (int i = nStart; i < nEnd; i++) {
        double x = set_x(mt64) * (xmax - xmin);
        particlesSpecies[i].x = x;
    }
}


void InitializeParticle::uniformForPositionY(
    int nStart, 
    int nEnd, 
    int seed, 
    std::vector<Particle>& particlesSpecies
)
{
    std::mt19937_64 mt64(seed);
    std::uniform_real_distribution<double> set_y(1e-20, 1.0 - 1e-20);

    for (int i = nStart; i < nEnd; i++) {
        double y = set_y(mt64) * (ymax - ymin);
        particlesSpecies[i].y = y;
    }
}


void InitializeParticle::maxwellDistributionForVelocity(
    double bulkVxSpecies, 
    double bulkVySpecies, 
    double bulkVzSpecies, 
    double vxThSpecies, 
    double vyThSpecies, 
    double vzThSpecies, 
    int nStart, 
    int nEnd, 
    int seed, 
    std::vector<Particle>& particlesSpecies
)
{
    std::mt19937_64 mt64Vx(seed);
    std::normal_distribution<double> set_vx(bulkVxSpecies, vxThSpecies);
    std::mt19937_64 mt64Vy(seed + 10000);
    std::normal_distribution<double> set_vy(bulkVySpecies, vyThSpecies);
    std::mt19937_64 mt64Vz(seed + 100000);
    std::normal_distribution<double> set_vz(bulkVzSpecies, vzThSpecies);

    for (int i = nStart; i < nEnd; i++) {
        double vx;
        double vy;
        double vz;

        while (true) {
            vx = set_vx(mt64Vx);
            vy = set_vy(mt64Vy);
            vz = set_vz(mt64Vz);

            if (vx * vx + vy * vy + vz * vz < c * c) break;
        }

        particlesSpecies[i].vx = vx;
        particlesSpecies[i].vy = vy;
        particlesSpecies[i].vz = vz;
        particlesSpecies[i].gamma = sqrt(1.0 + (vx * vx + vy * vy + vz * vz) / (c * c));
    }
}


void InitializeParticle::harrisForPositionY(
    int nStart, 
    int nEnd, 
    int seed, 
    std::vector<Particle>& particlesSpecies, 
    double sheatThickness
)
{
    std::mt19937_64 mt64(seed);
    std::uniform_real_distribution<double> set_y(1e-20, 1.0 - 1e-20);

    for (int i = nStart; i < nEnd; i++) {
        double random_value = set_y(mt64);
        double yCenter = 0.5 * (ymin + ymax);
        double yPosition = yCenter + sheatThickness * atanh(2.0 * random_value - 1.0);
        particlesSpecies[i].y = yPosition;
    }
}

