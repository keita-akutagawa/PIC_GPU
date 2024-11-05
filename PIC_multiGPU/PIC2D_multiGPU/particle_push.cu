#include "particle_push.hpp"


ParticlePush::ParticlePush(MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo)
{
}


void ParticlePush::pushVelocity(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron, 
    const thrust::device_vector<MagneticField>& B, 
    const thrust::device_vector<ElectricField>& E, 
    float dt
)
{
    MPI_Barrier(MPI_COMM_WORLD);
    pushVelocityOfOneSpecies(
        particlesIon, B, E, qIon, mIon, 
        mPIInfo.existNumIonPerProcs, dt
    );
    pushVelocityOfOneSpecies(
        particlesElectron, B, E, qElectron, mElectron, 
        mPIInfo.existNumElectronPerProcs, dt
    );
    MPI_Barrier(MPI_COMM_WORLD);
}


void ParticlePush::pushPosition(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron, 
    const float dt
)
{
    MPI_Barrier(MPI_COMM_WORLD);
    pushPositionOfOneSpecies(
        particlesIon, mPIInfo.existNumIonPerProcs, 
        mPIInfo.numForSendParticlesIonLeft, 
        mPIInfo.numForSendParticlesIonRight, 
        mPIInfo.numForSendParticlesIonDown, 
        mPIInfo.numForSendParticlesIonUp, 
        mPIInfo.numForSendParticlesIonCornerLeftDown, 
        mPIInfo.numForSendParticlesIonCornerRightDown, 
        mPIInfo.numForSendParticlesIonCornerLeftUp, 
        mPIInfo.numForSendParticlesIonCornerRightUp, 
        dt
    );
    pushPositionOfOneSpecies(
        particlesElectron, mPIInfo.existNumElectronPerProcs, 
        mPIInfo.numForSendParticlesElectronLeft, 
        mPIInfo.numForSendParticlesElectronRight, 
        mPIInfo.numForSendParticlesElectronDown, 
        mPIInfo.numForSendParticlesElectronUp, 
        mPIInfo.numForSendParticlesElectronCornerLeftDown, 
        mPIInfo.numForSendParticlesElectronCornerRightDown, 
        mPIInfo.numForSendParticlesElectronCornerLeftUp, 
        mPIInfo.numForSendParticlesElectronCornerRightUp, 
        dt
    );
    MPI_Barrier(MPI_COMM_WORLD);
}


//////////

__device__
ParticleField getParticleFields(
    const MagneticField* B,
    const ElectricField* E, 
    Particle& particle, 
    const int localNx, const int localNy, const int buffer, 
    const int localSizeX, const int localSizeY, 
    const float xminForProcs, const float xmaxForProcs, 
    const float yminForProcs, const float ymaxForProcs
)
{
    ParticleField particleField;

    float cx1, cx2; 
    int xIndex1, xIndex2;
    float xOverDx;
    float cy1, cy2; 
    int yIndex1, yIndex2;
    float yOverDy;

    xOverDx = (particle.x - xminForProcs + buffer * device_dx) / device_dx;
    yOverDy = (particle.y - yminForProcs + buffer * device_dy) / device_dy;

    xIndex1 = floorf(xOverDx);
    xIndex2 = xIndex1 + 1;
    xIndex2 = (xIndex2 == localSizeX) ? 0 : xIndex2;
    yIndex1 = floorf(yOverDy);
    yIndex2 = yIndex1 + 1;
    yIndex2 = (yIndex2 == localSizeY) ? 0 : yIndex2;

    cx1 = xOverDx - xIndex1;
    cx2 = 1.0f - cx1;
    cy1 = yOverDy - yIndex1;
    cy2 = 1.0f - cy1;

    particleField.bX += B[yIndex1 + localSizeY * xIndex1].bX * cx2 * cy2;
    particleField.bX += B[yIndex2 + localSizeY * xIndex1].bX * cx2 * cy1;
    particleField.bX += B[yIndex1 + localSizeY * xIndex2].bX * cx1 * cy2;
    particleField.bX += B[yIndex2 + localSizeY * xIndex2].bX * cx1 * cy1;

    particleField.bY += B[yIndex1 + localSizeY * xIndex1].bY * cx2 * cy2;
    particleField.bY += B[yIndex2 + localSizeY * xIndex1].bY * cx2 * cy1;
    particleField.bY += B[yIndex1 + localSizeY * xIndex2].bY * cx1 * cy2;
    particleField.bY += B[yIndex2 + localSizeY * xIndex2].bY * cx1 * cy1;

    particleField.bZ += B[yIndex1 + localSizeY * xIndex1].bZ * cx2 * cy2;
    particleField.bZ += B[yIndex2 + localSizeY * xIndex1].bZ * cx2 * cy1;
    particleField.bZ += B[yIndex1 + localSizeY * xIndex2].bZ * cx1 * cy2;
    particleField.bZ += B[yIndex2 + localSizeY * xIndex2].bZ * cx1 * cy1;

    particleField.eX += E[yIndex1 + localSizeY * xIndex1].eX * cx2 * cy2;
    particleField.eX += E[yIndex2 + localSizeY * xIndex1].eX * cx2 * cy1;
    particleField.eX += E[yIndex1 + localSizeY * xIndex2].eX * cx1 * cy2;
    particleField.eX += E[yIndex2 + localSizeY * xIndex2].eX * cx1 * cy1;

    particleField.eY += E[yIndex1 + localSizeY * xIndex1].eY * cx2 * cy2;
    particleField.eY += E[yIndex2 + localSizeY * xIndex1].eY * cx2 * cy1;
    particleField.eY += E[yIndex1 + localSizeY * xIndex2].eY * cx1 * cy2;
    particleField.eY += E[yIndex2 + localSizeY * xIndex2].eY * cx1 * cy1;

    particleField.eZ += E[yIndex1 + localSizeY * xIndex1].eZ * cx2 * cy2;
    particleField.eZ += E[yIndex2 + localSizeY * xIndex1].eZ * cx2 * cy1;
    particleField.eZ += E[yIndex1 + localSizeY * xIndex2].eZ * cx1 * cy2;
    particleField.eZ += E[yIndex2 + localSizeY * xIndex2].eZ * cx1 * cy1;

    return particleField;
}


__global__ void pushVelocityOfOneSpecies_kernel(
    Particle* particlesSpecies, const MagneticField* B, const ElectricField* E, 
    float q, float m, unsigned long long existNumSpecies, float dt, 
    const int localNx, const int localNy, const int buffer, 
    const int localSizeX, const int localSizeY, 
    const float xminForProcs, const float xmaxForProcs, 
    const float yminForProcs, const float ymaxForProcs
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
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

        vx = particlesSpecies[i].vx;
        vy = particlesSpecies[i].vy;
        vz = particlesSpecies[i].vz;
        gamma = particlesSpecies[i].gamma;

        particleField = getParticleFields(
            B, E, particlesSpecies[i], 
            localNx, localNy, buffer, 
            localSizeX, localSizeY, 
            xminForProcs, xmaxForProcs, 
            yminForProcs, ymaxForProcs
        );
        if (particlesSpecies[i].isExist == false) return;

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
    const float q, const float m, const unsigned long long existNumSpecies, 
    const float dt
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    pushVelocityOfOneSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data()), 
        q, m, existNumSpecies, dt, 
        mPIInfo.localNx, mPIInfo.localNy, mPIInfo.buffer, 
        mPIInfo.localSizeX, mPIInfo.localSizeY, 
        mPIInfo.xminForProcs, mPIInfo.xmaxForProcs, 
        mPIInfo.yminForProcs, mPIInfo.ymaxForProcs
    );
}


//////////

struct IsExistTransform
{
    __host__ __device__
    unsigned int operator()(const Particle& p) const {
        return p.isExist ? 1 : 0;
    }
};

__global__
void pushPositionOfOneSpecies_kernel(
    Particle* particlesSpecies, 
    const unsigned long long existNumSpecies, 
    unsigned int* numForSendParticlesSpeciesLeft, 
    unsigned int* numForSendParticlesSpeciesRight, 
    unsigned int* numForSendParticlesSpeciesDown, 
    unsigned int* numForSendParticlesSpeciesUp, 
    unsigned int* numForSendParticlesSpeciesCornerLeftDown, 
    unsigned int* numForSendParticlesSpeciesCornerRightDown, 
    unsigned int* numForSendParticlesSpeciesCornerLeftUp, 
    unsigned int* numForSendParticlesSpeciesCornerRightUp, 
    const float dt, 
    const float xminForProcs, const float xmaxForProcs, 
    const float yminForProcs, const float ymaxForProcs, 
    const int buffer
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {

        float vx, vy, vz, gamma;
        float xPast, yPast, zPast;
        float x, y, z;
        float dtOverGamma;

        vx = particlesSpecies[i].vx;
        vy = particlesSpecies[i].vy;
        vz = particlesSpecies[i].vz;
        gamma = particlesSpecies[i].gamma;
        xPast = particlesSpecies[i].x;
        yPast = particlesSpecies[i].y;
        zPast = particlesSpecies[i].z;

        dtOverGamma = dt / gamma;
        x = xPast + dtOverGamma * vx;
        y = yPast + dtOverGamma * vy;
        z = zPast + dtOverGamma * vz;

        particlesSpecies[i].x = x;
        particlesSpecies[i].y = y;
        particlesSpecies[i].z = z;

        float boundaryDeleteLeft  = xminForProcs - buffer * device_dx + device_EPS; 
        float boundaryDeleteRight = xmaxForProcs + buffer * device_dx - device_EPS; 
        float boundaryDeleteDown  = yminForProcs - buffer * device_dy + device_EPS; 
        float boundaryDeleteUp    = ymaxForProcs + buffer * device_dy - device_EPS;

        if (x < boundaryDeleteLeft) {
            particlesSpecies[i].isExist = false;
            return;
        }
        if (x > boundaryDeleteRight) {
            particlesSpecies[i].isExist = false;
            return;
        }
        if (y < boundaryDeleteDown) {
            particlesSpecies[i].isExist = false;
            return;
        }
        if (y > boundaryDeleteUp) {
            particlesSpecies[i].isExist = false;
            return;
        }


        float boundaryMPISendLeft  = xminForProcs + buffer * device_dx; 
        float boundaryMPISendRight = xmaxForProcs - buffer * device_dx; 
        float boundaryMPISendDown  = yminForProcs + buffer * device_dy; 
        float boundaryMPISendUp    = ymaxForProcs - buffer * device_dy; 
        
        if (xPast > boundaryMPISendLeft && x < boundaryMPISendLeft) {
            particlesSpecies[i].isMPISendLeft = true;
            atomicAdd(&(numForSendParticlesSpeciesLeft[0]), 1);
        }   
        if (xPast < boundaryMPISendRight && x > boundaryMPISendRight) {
            particlesSpecies[i].isMPISendRight = true;
            atomicAdd(&(numForSendParticlesSpeciesRight[0]), 1);
        }

        if (yPast > boundaryMPISendDown && y < boundaryMPISendDown) {
            particlesSpecies[i].isMPISendDown = true;
            atomicAdd(&(numForSendParticlesSpeciesDown[0]), 1);
            if (particlesSpecies[i].isMPISendLeft) {
                atomicAdd(&(numForSendParticlesSpeciesCornerLeftDown[0]), 1);
            }
            if (particlesSpecies[i].isMPISendRight) {
                atomicAdd(&(numForSendParticlesSpeciesCornerRightDown[0]), 1);
            }
        }   
        if (yPast < boundaryMPISendUp && y > boundaryMPISendUp) {
            particlesSpecies[i].isMPISendUp = true;
            atomicAdd(&(numForSendParticlesSpeciesUp[0]), 1);
            if (particlesSpecies[i].isMPISendLeft) {
                atomicAdd(&(numForSendParticlesSpeciesCornerLeftUp[0]), 1);
            }
            if (particlesSpecies[i].isMPISendRight) {
                atomicAdd(&(numForSendParticlesSpeciesCornerRightUp[0]), 1);
            }
        }
    }
}


void ParticlePush::pushPositionOfOneSpecies(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies, 
    unsigned int& numForSendParticlesSpeciesLeft, 
    unsigned int& numForSendParticlesSpeciesRight, 
    unsigned int& numForSendParticlesSpeciesDown, 
    unsigned int& numForSendParticlesSpeciesUp, 
    unsigned int& numForSendParticlesSpeciesCornerLeftDown, 
    unsigned int& numForSendParticlesSpeciesCornerRightDown, 
    unsigned int& numForSendParticlesSpeciesCornerLeftUp, 
    unsigned int& numForSendParticlesSpeciesCornerRightUp, 
    const float dt
)
{
    thrust::device_vector<unsigned int> countForSendParticlesSpeciesLeft(1, 0); 
    thrust::device_vector<unsigned int> countForSendParticlesSpeciesRight(1, 0); 
    thrust::device_vector<unsigned int> countForSendParticlesSpeciesDown(1, 0); 
    thrust::device_vector<unsigned int> countForSendParticlesSpeciesUp(1, 0); 
    thrust::device_vector<unsigned int> countForSendParticlesSpeciesCornerLeftDown(1, 0); 
    thrust::device_vector<unsigned int> countForSendParticlesSpeciesCornerRightDown(1, 0); 
    thrust::device_vector<unsigned int> countForSendParticlesSpeciesCornerLeftUp(1, 0); 
    thrust::device_vector<unsigned int> countForSendParticlesSpeciesCornerRightUp(1, 0); 

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    pushPositionOfOneSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        existNumSpecies, 
        thrust::raw_pointer_cast(countForSendParticlesSpeciesLeft.data()), 
        thrust::raw_pointer_cast(countForSendParticlesSpeciesRight.data()),  
        thrust::raw_pointer_cast(countForSendParticlesSpeciesDown.data()),  
        thrust::raw_pointer_cast(countForSendParticlesSpeciesUp.data()),  
        thrust::raw_pointer_cast(countForSendParticlesSpeciesCornerLeftDown.data()), 
        thrust::raw_pointer_cast(countForSendParticlesSpeciesCornerRightDown.data()),  
        thrust::raw_pointer_cast(countForSendParticlesSpeciesCornerLeftUp.data()),  
        thrust::raw_pointer_cast(countForSendParticlesSpeciesCornerRightUp.data()), 
        dt, 
        mPIInfo.xminForProcs, mPIInfo.xmaxForProcs, 
        mPIInfo.yminForProcs, mPIInfo.ymaxForProcs, 
        mPIInfo.buffer
    );
    cudaDeviceSynchronize();

    existNumSpecies = thrust::transform_reduce(
        particlesSpecies.begin(),
        particlesSpecies.end(),
        IsExistTransform(), 
        0,               
        thrust::plus<unsigned long long>()
    );
    cudaDeviceSynchronize();

    thrust::partition(
        particlesSpecies.begin(), particlesSpecies.end(), 
        [] __device__ (const Particle& p) { return p.isExist; }
    );
    cudaDeviceSynchronize();

    numForSendParticlesSpeciesLeft  = countForSendParticlesSpeciesLeft[0];
    numForSendParticlesSpeciesRight = countForSendParticlesSpeciesRight[0];
    numForSendParticlesSpeciesDown  = countForSendParticlesSpeciesDown[0];
    numForSendParticlesSpeciesUp    = countForSendParticlesSpeciesUp[0];
    numForSendParticlesSpeciesCornerLeftDown  = countForSendParticlesSpeciesCornerLeftDown[0];
    numForSendParticlesSpeciesCornerRightDown = countForSendParticlesSpeciesCornerRightDown[0];
    numForSendParticlesSpeciesCornerLeftUp    = countForSendParticlesSpeciesCornerLeftUp[0];
    numForSendParticlesSpeciesCornerRightUp   = countForSendParticlesSpeciesCornerRightUp[0];
}


