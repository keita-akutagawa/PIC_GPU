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

    void maxwellDistributionForVelocity(
        double bulkVxSpecies, 
        double bulkVySpecies, 
        double bulkVzSpecies, 
        double vThSpecies, 
        int nStart, 
        int nEnd, 
        int seed, 
        std::vector<Particle>& particlesSpecies
    );

private:

};

