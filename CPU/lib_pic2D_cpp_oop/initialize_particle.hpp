#include <vector>
#include "const.hpp"
#include "particle_struct.hpp"


class InitializeParticle
{
private:

public:
    void uniformForPositionX(
        int nStart, 
        int nEnd, 
        int seed, 
        std::vector<Particle>& particlesSpecies
    );

    void uniformForPositionY(
        int nStart, 
        int nEnd, 
        int seed, 
        std::vector<Particle>& particlesSpecies
    );

    void maxwellDistributionForVelocity(
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
    );

    void harrisForPositionY(
        int nStart, 
        int nEnd, 
        int seed, 
        std::vector<Particle>& particlesSpecies, 
        double sheatThickness
    );

private:

};

