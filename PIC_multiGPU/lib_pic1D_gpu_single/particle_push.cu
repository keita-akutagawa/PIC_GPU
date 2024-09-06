#include <cmath>
#include "particle_push.hpp"


void ParticlePush::pushVelocity(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron, 
    const thrust::device_vector<MagneticField>& B, 
    const thrust::device_vector<ElectricField>& E, 
    double dt, MPIInfo& mPIInfo
)
{
    double xminForProcs = xmin + (xmax - xmin) / mPIInfo.procs * mPIInfo.rank;
    double xmaxForProcs = xmin + (xmax - xmin) / mPIInfo.procs * (mPIInfo.rank + 1);

    pushVelocityOfOneSpecies(
        particlesIon, B, E, qIon, mIon, mPIInfo.existNumIonPerProcs, dt, xminForProcs, xmaxForProcs
    );
    pushVelocityOfOneSpecies(
        particlesElectron, B, E, qElectron, mElectron, mPIInfo.existNumElectronPerProcs, dt, xminForProcs, xmaxForProcs
    );
}


void ParticlePush::pushPosition(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron, 
    double dt, MPIInfo& mPIInfo
)
{
    pushPositionOfOneSpecies(
        particlesIon, mPIInfo.existNumIonPerProcs, dt
    );
    pushPositionOfOneSpecies(
        particlesElectron, mPIInfo.existNumElectronPerProcs, dt
    );
}


//////////

__device__
ParticleField getParticleFields(
    const MagneticField* B,
    const ElectricField* E, 
    const Particle& particle, 
    const double xminForProcs, const double xmaxForProcs
)
{
    ParticleField particleField;

    double cx1, cx2; 
    int xIndex1, xIndex2;

    double xOverDx;
    xOverDx = (particle.x - xminForProcs + device_dx) / device_dx;

    xIndex1 = floorf(xOverDx);
    xIndex2 = xIndex1 + 1;

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
    Particle* particlesSpecies, const MagneticField* B, const ElectricField* E, 
    double q, double m, int existNumSpecies, double dt, 
    const double xminForProcs, const double xmaxForProcs
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
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

        qOverMTimesDtOver2 = q / m * dt / 2.0f;
        tmp1OverC2 = 1.0f / (device_c * device_c);


        vx = particlesSpecies[i].vx;
        vy = particlesSpecies[i].vy;
        vz = particlesSpecies[i].vz;
        gamma = particlesSpecies[i].gamma;

        particleField = getParticleFields(B, E, particlesSpecies[i], xminForProcs, xmaxForProcs);
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
    double q, double m, int existNumSpecies, 
    double dt,  
    const double xminForProcs, const double xmaxForProcs
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    pushVelocityOfOneSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data()), 
        q, m, existNumSpecies, dt, 
        xminForProcs, xmaxForProcs
    );

    cudaDeviceSynchronize();
}


//////////

__global__
void pushPositionOfOneSpecies_kernel(
    Particle* particlesSpecies, int existNumSpecies, double dt
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
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
    int existNumSpecies, 
    double dt
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    pushPositionOfOneSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        existNumSpecies, dt
    );

    cudaDeviceSynchronize();
}


