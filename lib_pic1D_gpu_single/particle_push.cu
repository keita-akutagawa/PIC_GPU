#include <cmath>
#include "particle_push.hpp"


void ParticlePush::pushVelocity(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron, 
    const thrust::device_vector<MagneticField>& B, 
    const thrust::device_vector<ElectricField>& E, 
    float dt
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
    float dt
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
    const MagneticField* B,
    const ElectricField* E, 
    const Particle& particle
)
{
    ParticleField particleField;

    float cx1, cx2; 
    int xIndex1, xIndex2;

    float xOverDx;
    xOverDx = particle.x / device_dx;

    xIndex1 = std::floor(xOverDx);
    xIndex2 = xIndex1 + 1;
    xIndex2 = (xIndex2 == device_nx) ? 0 : xIndex2;

    cx1 = xOverDx - xIndex1;
    cx2 = 1.0f - cx1;

    particleField.bX += B[xIndex1].bX * cx2;
    particleField.bX += B[xIndex2].bX * cx1;

    particleField.bY += B[xIndex1].bY * cx2;
    particleField.bY += B[xIndex2].bY * cx1;

    particleField.bZ += B[xIndex1].bZ * cx2;
    particleField.bZ += B[xIndex2].bZ * cx1;

    particleField.eX += E[xIndex1].eX * cx2;
    particleField.eX += E[xIndex2].eX * cx1;

    particleField.eY += E[xIndex1].eY * cx2;
    particleField.eY += E[xIndex2].eY * cx1;

    particleField.eZ += E[xIndex1].eZ * cx2;
    particleField.eZ += E[xIndex2].eZ * cx1;


    return particleField;
}


__global__
void pushVelocityOfOneSpecies_kernel(
    Particle* particlesSpecies, const MagneticField* magneticField, const ElectricField* electricField, 
    float q, float m, int totalNumSpecies, float dt
)
{
    float qOverMTimesDtOver2;
    float tmpForT, tmpForS, tmp1OverC2;
    float vx, vy, vz, gamma;
    float tx, ty, tz;
    float sx, sy, sz;
    float vxMinus, vyMinus, vzMinus;
    float vx0, vy0, vz0;
    float vxPlus, vyPlus, vzPlus; 
    float bx, by, bz;
    float ex, ey, ez;
    ParticleField particleField;

    qOverMTimesDtOver2 = q / m * dt / 2.0f;
    tmp1OverC2 = 1.0f / (device_c * device_c);

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < totalNumSpecies) {
        vx = particlesSpecies[i].vx;
        vy = particlesSpecies[i].vy;
        vz = particlesSpecies[i].vz;
        gamma = particlesSpecies[i].gamma;

        particleField = getParticleFields(magneticField, electricField, particlesSpecies[i]);
        bx = particleField.bX;
        by = particleField.bY;
        bz = particleField.bZ; 
        ex = particleField.eX;
        ey = particleField.eY; 
        ez = particleField.eZ;

        tmpForT = qOverMTimesDtOver2 / gamma;
        tx = tmpForT * bx;
        ty = tmpForT * by;
        tz = tmpForT * bz;

        tmpForS = 2.0f / (1.0f + tx * tx + ty * ty + tz * tz);
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
        gamma = sqrt(1.0f + (vx * vx + vy * vy + vz * vz) * tmp1OverC2);

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
    float q, float m, int totalNumSpecies, 
    float dt
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
    Particle* particlesSpecies, int totalNumSpecies, float dt
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < totalNumSpecies) {
        float vx, vy, vz, gamma;
        float x, y, z;
        float dtOverGamma;

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
    float dt
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((totalNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    pushPositionOfOneSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        totalNumSpecies, dt
    );

    cudaDeviceSynchronize();
}


