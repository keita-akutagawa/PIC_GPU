#include <cmath>
#include "particle_push.hpp"


void ParticlePush::pushVelocity(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron, 
    const thrust::device_vector<MagneticField>& B, 
    const thrust::device_vector<ElectricField>& E, 
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
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron, 
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


//////////

__device__
ParticleField getParticleFields(
    const MagneticField* magneticField,
    const ElectricField* electricField, 
    const Particle& particle
)
{
    ParticleField particleField;

    double cx1, cx2; 
    int xIndex1, xIndex2;

    double xOverDx;
    xOverDx = particle.x / device_dx;

    xIndex1 = std::floor(xOverDx);
    xIndex2 = xIndex1 + 1;
    xIndex2 = (xIndex2 == device_nx) ? 0 : xIndex2;

    cx1 = xOverDx - xIndex1;
    cx2 = 1.0 - cx1;

    particleField.bx += magneticField[xIndex1].bX * cx2;
    particleField.bx += magneticField[xIndex2].bX * cx1;

    particleField.by += magneticField[xIndex1].bY * cx2;
    particleField.by += magneticField[xIndex2].bY * cx1;

    particleField.bz += magneticField[xIndex1].bZ * cx2;
    particleField.bz += magneticField[xIndex2].bZ * cx1;

    particleField.ex += electricField[xIndex1].eX * cx2;
    particleField.ex += electricField[xIndex2].eX * cx1;

    particleField.ey += electricField[xIndex1].eY * cx2;
    particleField.ey += electricField[xIndex2].eY * cx1;

    particleField.ez += electricField[xIndex1].eZ * cx2;
    particleField.ez += electricField[xIndex2].eZ * cx1;


    return particleField;
}


__global__
void pushVelocityOfOneSpecies_kernel(
    Particle* particlesSpecies, const MagneticField* magneticField, const ElectricField* electricField, 
    double q, double m, int totalNumSpecies, double dt
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
    tmp1OverC2 = 1.0 / (device_c * device_c);

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < totalNumSpecies) {
        vx = particlesSpecies[i].vx;
        vy = particlesSpecies[i].vy;
        vz = particlesSpecies[i].vz;
        gamma = particlesSpecies[i].gamma;

        particleField = getParticleFields(magneticField, electricField, particlesSpecies[i]);
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


void ParticlePush::pushVelocityOfOneSpecies(
    thrust::device_vector<Particle>& particlesSpecies, 
    const thrust::device_vector<MagneticField>& B,
    const thrust::device_vector<ElectricField>& E, 
    double q, double m, int totalNumSpecies, 
    double dt
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((totalNumParticles + threadsPerBlock.x - 1) / threadsPerBlock.x);

    pushVelocityOfOneSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data()), 
        q, m, totalNumSpecies, dt
    );

    cudaDeviceSynchronize();
}


//////////

__global__
void pushPositionOfOneSpecies_kernel(
    Particle* particlesSpecies, int totalNumSpecies, double dt
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < totalNumSpecies) {
        double vx, vy, vz, gamma;
        double x, y, z;
        double dtOverGamma;

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


void ParticlePush::pushPositionOfOneSpecies(
    thrust::device_vector<Particle>& particlesSpecies, 
    int totalNumSpecies, 
    double dt
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((totalNumParticles + threadsPerBlock.x - 1) / threadsPerBlock.x);

    pushPositionOfOneSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        totalNumSpecies, dt
    );

    cudaDeviceSynchronize();
}


