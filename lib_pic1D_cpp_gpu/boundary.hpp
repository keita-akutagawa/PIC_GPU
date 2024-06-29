#include <vector>
#include "const.hpp"
#include "particle_struct.hpp"


class Boundary
{
private:

public:

    void periodicBoundaryParticleX(
        std::vector<Particle>& particlesIon,
        std::vector<Particle>& particlesElectron
    );
    void conductingWallBoundaryParticleX(
        std::vector<Particle>& particlesIon,
        std::vector<Particle>& particlesElectron
    );

    void periodicBoundaryBX();
    void conductingWallBoundaryBX();

    void periodicBoundaryEX();
    void conductingWallBoundaryEX();

    void periodicBoundaryCurrentX();
    void conductingWallBoundaryCurrentX();

private:

};


