#include <cmath>
#include "particle_push.hpp"


void ParticlePush::pushVelocity(
    std::vector<Particle>& particlesIon, 
    std::vector<Particle>& particlesElectron, 
    const std::vector<std::vector<double>>& B, 
    const std::vector<std::vector<double>>& E, 
    double dt
)
{
    pushVelocityOfOneSpecies(
        particlesIon, B, E, qIon, mIon, totalNumIon, dt
    );
    pushVelocityOfOneSpecies(
        particlesElectron, B, E, qElectron, mElectron, totalNumElectron, dt
    );
}


void ParticlePush::pushPosition(
    std::vector<Particle>& particlesIon, 
    std::vector<Particle>& particlesElectron, 
    double dt
)
{
    pushPositionOfOneSpecies(
        particlesIon, totalNumIon, dt
    );
    pushPositionOfOneSpecies(
        particlesElectron, totalNumElectron, dt
    );
}


void ParticlePush::pushVelocityOfOneSpecies(
    std::vector<Particle>& particlesSpecies, 
    const std::vector<std::vector<double>>& B, 
    const std::vector<std::vector<double>>& E,     
    double q, double m, int totalNumSpecies, 
    double dt
)
{
    double qOverMTimesDtOver2;
    double tmpForT, tmpForS, tmp1OverC2;
    double vx, vy, vz, gamma;
    double tx, ty, tz;
    double sx, sy, sz;
    double vxMinus, vyMinus, vzMinus;
    double vx0, vy0, vz0;
    double vxPlus, vyPlus, vzPlus; 
    double bx, by, bz;
    double ex, ey, ez;
    ParticleField particleField;

    qOverMTimesDtOver2 = q / m * dt / 2.0;
    tmp1OverC2 = 1.0 / (c * c);

    for (int i = 0; i < totalNumSpecies; i++) {

        vx = particlesSpecies[i].vx;
        vy = particlesSpecies[i].vy;
        vz = particlesSpecies[i].vz;
        gamma = particlesSpecies[i].gamma;

        particleField = getParticleFields(B, E, particlesSpecies[i]);
        bx = particleField.bx;
        by = particleField.by;
        bz = particleField.bz; 
        ex = particleField.ex;
        ey = particleField.ey; 
        ez = particleField.ez;


        tmpForT = qOverMTimesDtOver2 / gamma;
        tx = tmpForT * bx;
        ty = tmpForT * by;
        tz = tmpForT * bz;

        tmpForS = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz);
        sx = tmpForS * tx;
        sy = tmpForS * ty;
        sz = tmpForS * tz;

        vxMinus = vx + qOverMTimesDtOver2 * ex;
        vyMinus = vy + qOverMTimesDtOver2 * ey;
        vzMinus = vz + qOverMTimesDtOver2 * ez;

        vx0 = vxMinus + (vyMinus * tz - vzMinus * ty);
        vy0 = vyMinus + (vzMinus * tx - vxMinus * tz);
        vz0 = vzMinus + (vxMinus * ty - vyMinus * tx);

        vxPlus = vxMinus + (vy0 * sz - vz0 * sy);
        vyPlus = vyMinus + (vz0 * sx - vx0 * sz);
        vzPlus = vzMinus + (vx0 * sy - vy0 * sx);

        vx = vxPlus + qOverMTimesDtOver2 * ex;
        vy = vyPlus + qOverMTimesDtOver2 * ey;
        vz = vzPlus + qOverMTimesDtOver2 * ez;
        gamma = sqrt(1.0 + (vx * vx + vy * vy + vz * vz) * tmp1OverC2);

        particlesSpecies[i].vx = vx;
        particlesSpecies[i].vy = vy;
        particlesSpecies[i].vz = vz;
        particlesSpecies[i].gamma = gamma;
    } 
}


inline ParticleField ParticlePush::getParticleFields(
    const std::vector<std::vector<double>>& B, 
    const std::vector<std::vector<double>>& E, 
    const Particle& particle
)
{
    // 呼び出されるごとにコンストラクタで0.0に初期化されるはず
    ParticleField particleField;

    double cx1, cx2, xIndex1, xIndex2;
    double xOverDx;

    xOverDx = particle.x / dx;

    xIndex1 = std::floor(xOverDx);
    xIndex2 = xIndex1 + 1;
    xIndex2 = (xIndex2 == nx) ? 0 : xIndex2;

    cx1 = xOverDx - xIndex1;
    cx2 = 1.0 - cx1;


    particleField.bx += B[0][xIndex1] * cx2;
    particleField.bx += B[0][xIndex2] * cx1;

    particleField.by += B[1][xIndex1] * cx2;
    particleField.by += B[1][xIndex2] * cx1;

    particleField.bz += B[2][xIndex1] * cx2;
    particleField.bz += B[2][xIndex2] * cx1;

    particleField.ex += E[0][xIndex1] * cx2;
    particleField.ex += E[0][xIndex2] * cx1;

    particleField.ey += E[1][xIndex1] * cx2;
    particleField.ey += E[1][xIndex2] * cx1;

    particleField.ez += E[2][xIndex1] * cx2;
    particleField.ez += E[2][xIndex2] * cx1;


    return particleField;
}


void ParticlePush::pushPositionOfOneSpecies(
    std::vector<Particle>& particlesSpecies, 
    int totalNumSpecies, 
    double dt
)
{
    double vx, vy, vz, gamma;
    double x, y, z;
    double dtOverGamma;

    for (int i = 0; i < totalNumSpecies; i++) {
        vx = particlesSpecies[i].vx;
        vy = particlesSpecies[i].vy;
        vz = particlesSpecies[i].vz;
        gamma = particlesSpecies[i].gamma;
        x = particlesSpecies[i].x;
        y = particlesSpecies[i].y;
        z = particlesSpecies[i].z;

        dtOverGamma = dt / gamma;
        x += dtOverGamma * vx;
        y += dtOverGamma * vy;
        z += dtOverGamma * vz;

        particlesSpecies[i].x = x;
        particlesSpecies[i].y = y;
        particlesSpecies[i].z = z;
    }
}


