#include "boundary.hpp"


Boundary::Boundary(MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo), 

      sendParticlesSpeciesLeftToRight(numberDensityIon * 10 * mPIInfo.localSizeY), 
      sendParticlesSpeciesRightToLeft(numberDensityIon * 10 * mPIInfo.localSizeY), 
      recvParticlesSpeciesLeftToRight(numberDensityIon * 10 * mPIInfo.localSizeY), 
      recvParticlesSpeciesRightToLeft(numberDensityIon * 10 * mPIInfo.localSizeY), 

      sendParticlesSpeciesUpToDown(numberDensityIon * mPIInfo.localSizeX * 10), 
      sendParticlesSpeciesDownToUp(numberDensityIon * mPIInfo.localSizeX * 10), 
      recvParticlesSpeciesUpToDown(numberDensityIon * mPIInfo.localSizeX * 10), 
      recvParticlesSpeciesDownToUp(numberDensityIon * mPIInfo.localSizeX * 10), 

      host_sendParticlesSpeciesLeftToRight(numberDensityIon * 10 * mPIInfo.localSizeY), 
      host_sendParticlesSpeciesRightToLeft(numberDensityIon * 10 * mPIInfo.localSizeY), 
      host_recvParticlesSpeciesLeftToRight(numberDensityIon * 10 * mPIInfo.localSizeY), 
      host_recvParticlesSpeciesRightToLeft(numberDensityIon * 10 * mPIInfo.localSizeY), 

      host_sendParticlesSpeciesUpToDown(numberDensityIon * mPIInfo.localSizeX * 10), 
      host_sendParticlesSpeciesDownToUp(numberDensityIon * mPIInfo.localSizeX * 10), 
      host_recvParticlesSpeciesUpToDown(numberDensityIon * mPIInfo.localSizeX * 10), 
      host_recvParticlesSpeciesDownToUp(numberDensityIon * mPIInfo.localSizeX * 10)
{
}


struct IsExistTransform
{
    __host__ __device__
    unsigned long long operator()(const Particle& p) const {
        return p.isExist ? 1 : 0;
    }
};


void Boundary::periodicBoundaryParticleXY(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron
)
{   
    periodicBoundaryParticleOfOneSpeciesX(
        particlesIon,
        mPIInfo.existNumIonPerProcs, 
        mPIInfo.countForSendIonLeftToRight, 
        mPIInfo.countForSendIonRightToLeft, 
        mPIInfo.countForRecvIonLeftToRight, 
        mPIInfo.countForRecvIonRightToLeft
    );
    periodicBoundaryParticleOfOneSpeciesX(
        particlesElectron, 
        mPIInfo.existNumElectronPerProcs,  
        mPIInfo.countForSendElectronLeftToRight, 
        mPIInfo.countForSendElectronRightToLeft, 
        mPIInfo.countForRecvElectronLeftToRight, 
        mPIInfo.countForRecvElectronRightToLeft
    );
    MPI_Barrier(MPI_COMM_WORLD);

    periodicBoundaryParticleOfOneSpeciesY(
        particlesIon,
        mPIInfo.existNumIonPerProcs, 
        mPIInfo.countForSendIonDownToUp, 
        mPIInfo.countForSendIonUpToDown, 
        mPIInfo.countForRecvIonDownToUp, 
        mPIInfo.countForRecvIonUpToDown
    );
    periodicBoundaryParticleOfOneSpeciesY(
        particlesElectron, 
        mPIInfo.existNumElectronPerProcs, 
        mPIInfo.countForSendElectronDownToUp, 
        mPIInfo.countForSendElectronUpToDown, 
        mPIInfo.countForRecvElectronDownToUp, 
        mPIInfo.countForRecvElectronUpToDown
    );
    MPI_Barrier(MPI_COMM_WORLD);
}


__global__ void periodicBoundaryParticleX_kernel(
    Particle* particlesSpecies, 
    Particle* sendParticlesSpeciesLeftToRight, 
    Particle* sendParticlesSpeciesRightToLeft, 
    unsigned long long* countForSendSpeciesLeftToRight, 
    unsigned long long* countForSendSpeciesRightToLeft, 
    const unsigned long long existNumSpecies, 
    const float xminForProcs, const float xmaxForProcs, 
    const int buffer
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (particlesSpecies[i].x - (xminForProcs - buffer * device_dx) < device_EPS) {
            particlesSpecies[i].isExist = false;
        }

        if (particlesSpecies[i].x - (xmaxForProcs + buffer * device_dx) > -device_EPS) {
            particlesSpecies[i].isExist = false;
        }

        if (particlesSpecies[i].isMPISendLeftToRight) {
            unsigned long long particleIndex = atomicAdd(&(countForSendSpeciesLeftToRight[0]), 1);
            particlesSpecies[i].isMPISendLeftToRight = false;
            Particle sendParticle = particlesSpecies[i];
            if (sendParticle.x < device_xmin + buffer * device_dx) {
                sendParticle.x += device_xmax - device_EPS;
            }
            sendParticlesSpeciesLeftToRight[particleIndex] = sendParticle;
        }

        if (particlesSpecies[i].isMPISendRightToLeft) {
            unsigned long long particleIndex = atomicAdd(&(countForSendSpeciesRightToLeft[0]), 1);
            particlesSpecies[i].isMPISendRightToLeft = false;
            Particle sendParticle = particlesSpecies[i];
            if (sendParticle.x > device_xmax - buffer * device_dx) {
                sendParticle.x -= device_xmax + device_EPS;
            }
            sendParticlesSpeciesRightToLeft[particleIndex] = sendParticle;
        }
    }
}

void Boundary::periodicBoundaryParticleOfOneSpeciesX(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies, 
    unsigned long long& countForSendSpeciesLeftToRight, 
    unsigned long long& countForSendSpeciesRightToLeft, 
    unsigned long long& countForRecvSpeciesLeftToRight, 
    unsigned long long& countForRecvSpeciesRightToLeft
)
{
    countForSendSpeciesLeftToRight = 0;
    countForSendSpeciesRightToLeft = 0;
    countForRecvSpeciesLeftToRight = 0;
    countForRecvSpeciesRightToLeft = 0;

    thrust::device_vector<unsigned long long> device_countForSendSpeciesLeftToRight(1, 0); 
    thrust::device_vector<unsigned long long> device_countForSendSpeciesRightToLeft(1, 0); 

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    periodicBoundaryParticleX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesLeftToRight.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesRightToLeft.data()), 
        thrust::raw_pointer_cast(device_countForSendSpeciesLeftToRight.data()), 
        thrust::raw_pointer_cast(device_countForSendSpeciesRightToLeft.data()), 
        existNumSpecies, 
        mPIInfo.xminForProcs, mPIInfo.xmaxForProcs, 
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

    host_sendParticlesSpeciesLeftToRight = sendParticlesSpeciesLeftToRight;
    host_sendParticlesSpeciesRightToLeft = sendParticlesSpeciesRightToLeft;
    countForSendSpeciesLeftToRight = device_countForSendSpeciesLeftToRight[0];
    countForSendSpeciesRightToLeft = device_countForSendSpeciesRightToLeft[0];

    sendrecv_particle_x(
        host_sendParticlesSpeciesLeftToRight, 
        host_sendParticlesSpeciesRightToLeft, 
        host_recvParticlesSpeciesLeftToRight, 
        host_recvParticlesSpeciesRightToLeft, 
        countForSendSpeciesLeftToRight, 
        countForSendSpeciesRightToLeft, 
        countForRecvSpeciesLeftToRight, 
        countForRecvSpeciesRightToLeft, 
        mPIInfo
    );

    recvParticlesSpeciesLeftToRight = host_recvParticlesSpeciesLeftToRight;
    recvParticlesSpeciesRightToLeft = host_recvParticlesSpeciesRightToLeft;

    for (unsigned long long i = 0; i < countForRecvSpeciesLeftToRight; i++) {
        particlesSpecies[existNumSpecies + i] = recvParticlesSpeciesLeftToRight[i];
    }
    existNumSpecies += countForRecvSpeciesLeftToRight;
    for (unsigned long long i = 0; i < countForRecvSpeciesRightToLeft; i++) {
        particlesSpecies[existNumSpecies + i] = recvParticlesSpeciesRightToLeft[i];
    }
    existNumSpecies += countForRecvSpeciesRightToLeft;
}


__global__ void periodicBoundaryParticleY_kernel(
    Particle* particlesSpecies, 
    Particle* sendParticlesSpeciesUpToDown, 
    Particle* sendParticlesSpeciesDownToUp, 
    unsigned long long* countForSendSpeciesUpToDown, 
    unsigned long long* countForSendSpeciesDownToUp, 
    const unsigned long long existNumSpecies, 
    const float yminForProcs, const float ymaxForProcs, 
    const int buffer
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (particlesSpecies[i].y - (yminForProcs - buffer * device_dy) < device_EPS) {
            particlesSpecies[i].isExist = false;
        }

        if (particlesSpecies[i].y - (ymaxForProcs + buffer * device_dy) > -device_EPS) {
            particlesSpecies[i].isExist = false;
        }

        if (particlesSpecies[i].isMPISendDownToUp) {
            unsigned long long particleIndex = atomicAdd(&(countForSendSpeciesDownToUp[0]), 1);
            particlesSpecies[i].isMPISendDownToUp = false;
            Particle sendParticle = particlesSpecies[i];
            if (sendParticle.y < device_ymin + buffer * device_dy) {
                sendParticle.y += device_ymax - device_EPS;
            }
            sendParticlesSpeciesDownToUp[particleIndex] = sendParticle;
        }

        if (particlesSpecies[i].isMPISendUpToDown) {
            unsigned long long particleIndex = atomicAdd(&(countForSendSpeciesUpToDown[0]), 1);
            particlesSpecies[i].isMPISendUpToDown = false;
            Particle sendParticle = particlesSpecies[i];
            if (sendParticle.y > device_ymax - buffer * device_dy) {
                sendParticle.y -= device_ymax + device_EPS;
            }
            sendParticlesSpeciesUpToDown[particleIndex] = sendParticle;
        }
    }
}

void Boundary::periodicBoundaryParticleOfOneSpeciesY(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies, 
    unsigned long long& countForSendSpeciesDownToUp, 
    unsigned long long& countForSendSpeciesUpToDown, 
    unsigned long long& countForRecvSpeciesDownToUp, 
    unsigned long long& countForRecvSpeciesUpToDown
)
{
    countForSendSpeciesUpToDown = 0;
    countForSendSpeciesDownToUp = 0;
    countForRecvSpeciesUpToDown = 0;
    countForRecvSpeciesDownToUp = 0;

    thrust::device_vector<unsigned long long> device_countForSendSpeciesUpToDown(1, 0); 
    thrust::device_vector<unsigned long long> device_countForSendSpeciesDownToUp(1, 0); 

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    periodicBoundaryParticleY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesUpToDown.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesDownToUp.data()), 
        thrust::raw_pointer_cast(device_countForSendSpeciesUpToDown.data()), 
        thrust::raw_pointer_cast(device_countForSendSpeciesDownToUp.data()), 
        existNumSpecies, 
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

    host_sendParticlesSpeciesUpToDown = sendParticlesSpeciesUpToDown;
    host_sendParticlesSpeciesDownToUp = sendParticlesSpeciesDownToUp;
    countForSendSpeciesUpToDown = device_countForSendSpeciesUpToDown[0];
    countForSendSpeciesDownToUp = device_countForSendSpeciesDownToUp[0];

    sendrecv_particle_y(
        host_sendParticlesSpeciesUpToDown, 
        host_sendParticlesSpeciesDownToUp, 
        host_recvParticlesSpeciesUpToDown, 
        host_recvParticlesSpeciesDownToUp, 
        countForSendSpeciesUpToDown, 
        countForSendSpeciesDownToUp, 
        countForRecvSpeciesUpToDown, 
        countForRecvSpeciesDownToUp, 
        mPIInfo
    );

    recvParticlesSpeciesUpToDown = host_recvParticlesSpeciesUpToDown;
    recvParticlesSpeciesDownToUp = host_recvParticlesSpeciesDownToUp;

    for (unsigned long long i = 0; i < countForRecvSpeciesUpToDown; i++) {
        particlesSpecies[existNumSpecies + i] = recvParticlesSpeciesUpToDown[i];
    }
    existNumSpecies += countForRecvSpeciesUpToDown;
    for (unsigned long long i = 0; i < countForRecvSpeciesDownToUp; i++) {
        particlesSpecies[existNumSpecies + i] = recvParticlesSpeciesDownToUp[i];
    }
    existNumSpecies += countForRecvSpeciesDownToUp;

}

