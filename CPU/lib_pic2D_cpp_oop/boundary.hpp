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
    void periodicBoundaryParticleY(
        std::vector<Particle>& particlesIon,
        std::vector<Particle>& particlesElectron
    );
    void conductingWallBoundaryParticleX(
        std::vector<Particle>& particlesIon,
        std::vector<Particle>& particlesElectron
    );
    void conductingWallBoundaryParticleY(
        std::vector<Particle>& particlesIon,
        std::vector<Particle>& particlesElectron
    );

    void periodicBoundaryBX();
    void periodicBoundaryBY();
    void conductingWallBoundaryBX(std::vector<std::vector<std::vector<double>>>& B);
    void conductingWallBoundaryBY(std::vector<std::vector<std::vector<double>>>& B);
    void symmetricWallBoundaryBX(std::vector<std::vector<std::vector<double>>>& B);
    void symmetricWallBoundaryBY(std::vector<std::vector<std::vector<double>>>& B);

    void periodicBoundaryEX();
    void periodicBoundaryEY();
    void conductingWallBoundaryEX(std::vector<std::vector<std::vector<double>>>& E);
    void conductingWallBoundaryEY(std::vector<std::vector<std::vector<double>>>& E);
    void symmetricWallBoundaryEX(std::vector<std::vector<std::vector<double>>>& E);
    void symmetricWallBoundaryEY(std::vector<std::vector<std::vector<double>>>& E);

    void periodicBoundaryCurrentX();
    void periodicBoundaryCurrentY();
    void conductingWallBoundaryCurrentX(std::vector<std::vector<std::vector<double>>>& current);
    void conductingWallBoundaryCurrentY(std::vector<std::vector<std::vector<double>>>& current);
    void symmetricWallBoundaryCurrentX(std::vector<std::vector<std::vector<double>>>& current);
    void symmetricWallBoundaryCurrentY(std::vector<std::vector<std::vector<double>>>& current);

private:

};


