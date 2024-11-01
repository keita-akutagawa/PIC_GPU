#include "boundary.hpp"


void Boundary::boundaryForInitializeParticle(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron
)
{
    boundaryForInitializeParticleOfOneSpeciesX(
        particlesIon, 
        mPIInfo.existNumIonPerProcs, 
        mPIInfo.countForSendIonLeftToRight, 
        mPIInfo.countForSendIonRightToLeft, 
        mPIInfo.countForRecvIonLeftToRight, 
        mPIInfo.countForRecvIonRightToLeft
    ); 
    boundaryForInitializeParticleOfOneSpeciesX(
        particlesElectron, 
        mPIInfo.existNumElectronPerProcs,  
        mPIInfo.countForSendElectronLeftToRight, 
        mPIInfo.countForSendElectronRightToLeft, 
        mPIInfo.countForRecvElectronLeftToRight, 
        mPIInfo.countForRecvElectronRightToLeft
    ); 

    MPI_Barrier(MPI_COMM_WORLD);

    boundaryForInitializeParticleOfOneSpeciesY(
        particlesIon, 
        mPIInfo.existNumIonPerProcs, 
        mPIInfo.countForSendIonDownToUp, 
        mPIInfo.countForSendIonUpToDown, 
        mPIInfo.countForRecvIonDownToUp, 
        mPIInfo.countForRecvIonUpToDown
    ); 
    boundaryForInitializeParticleOfOneSpeciesY(
        particlesElectron, 
        mPIInfo.existNumElectronPerProcs,  
        mPIInfo.countForSendElectronDownToUp, 
        mPIInfo.countForSendElectronUpToDown, 
        mPIInfo.countForRecvElectronDownToUp, 
        mPIInfo.countForRecvElectronUpToDown
    ); 

    MPI_Barrier(MPI_COMM_WORLD);
}


__global__ void boundaryForInitializeX_kernel(
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
        if (xminForProcs < particlesSpecies[i].x && particlesSpecies[i].x < xminForProcs + buffer * device_dx) {
            unsigned long long particleIndex = atomicAdd(&(countForSendSpeciesLeftToRight[0]), 1);
            Particle sendParticle = particlesSpecies[i];
            if (sendParticle.x < device_xmin + buffer * device_dx) {
                sendParticle.x += device_xmax;
            }
            sendParticlesSpeciesLeftToRight[particleIndex] = sendParticle;
        }

        if (xmaxForProcs - buffer * device_dx < particlesSpecies[i].x && particlesSpecies[i].x < xmaxForProcs) {
            unsigned long long particleIndex = atomicAdd(&(countForSendSpeciesRightToLeft[0]), 1);
            Particle sendParticle = particlesSpecies[i];
            if (device_xmax - buffer * device_dx < sendParticle.x) {
                sendParticle.x -= device_xmax;
            }
            sendParticlesSpeciesRightToLeft[particleIndex] = sendParticle;
        }
    }
}

void Boundary::boundaryForInitializeParticleOfOneSpeciesX(
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

    boundaryForInitializeX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
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

    for (unsigned long long i = 0; i < countForRecvSpeciesLeftToRight; i++) {
        particlesSpecies[existNumSpecies + i] = host_recvParticlesSpeciesLeftToRight[i];
    }
    existNumSpecies += countForRecvSpeciesLeftToRight;
    for (unsigned long long i = 0; i < countForRecvSpeciesRightToLeft; i++) {
        particlesSpecies[existNumSpecies + i] = host_recvParticlesSpeciesRightToLeft[i];
    }
    existNumSpecies += countForRecvSpeciesRightToLeft;
}


__global__ void boundaryForInitializeY_kernel(
    Particle* particlesSpecies, 
    Particle* sendParticlesSpeciesDownToUp, 
    Particle* sendParticlesSpeciesUpToDown, 
    unsigned long long* countForSendSpeciesDownToUp, 
    unsigned long long* countForSendSpeciesUpToDown, 
    const unsigned long long existNumSpecies, 
    const float yminForProcs, const float ymaxForProcs, 
    const int buffer
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (yminForProcs < particlesSpecies[i].y && particlesSpecies[i].y < yminForProcs + buffer * device_dy) {
            unsigned long long particleIndex = atomicAdd(&(countForSendSpeciesDownToUp[0]), 1);
            Particle sendParticle = particlesSpecies[i];
            if (sendParticle.y < device_ymin + buffer * device_dy) {
                sendParticle.y += device_ymax;
            }
            sendParticlesSpeciesDownToUp[particleIndex] = sendParticle;
        }

        if (ymaxForProcs - buffer * device_dy < particlesSpecies[i].y && particlesSpecies[i].y < ymaxForProcs) {
            unsigned long long particleIndex = atomicAdd(&(countForSendSpeciesUpToDown[0]), 1);
            Particle sendParticle = particlesSpecies[i];
            if (device_ymax - buffer * device_dy < sendParticle.y) {
                sendParticle.y -= device_ymax;
            }
            sendParticlesSpeciesUpToDown[particleIndex] = sendParticle;
        }
    }
}

void Boundary::boundaryForInitializeParticleOfOneSpeciesY(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies, 
    unsigned long long& countForSendSpeciesDownToUp, 
    unsigned long long& countForSendSpeciesUpToDown, 
    unsigned long long& countForRecvSpeciesDownToUp, 
    unsigned long long& countForRecvSpeciesUpToDown
)
{
    countForSendSpeciesDownToUp = 0;
    countForSendSpeciesUpToDown = 0;
    countForRecvSpeciesDownToUp = 0;
    countForRecvSpeciesUpToDown = 0;

    thrust::device_vector<unsigned long long> device_countForSendSpeciesDownToUp(1, 0); 
    thrust::device_vector<unsigned long long> device_countForSendSpeciesUpToDown(1, 0);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    boundaryForInitializeY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesDownToUp.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesUpToDown.data()), 
        thrust::raw_pointer_cast(device_countForSendSpeciesDownToUp.data()), 
        thrust::raw_pointer_cast(device_countForSendSpeciesUpToDown.data()), 
        existNumSpecies, 
        mPIInfo.yminForProcs, mPIInfo.ymaxForProcs, 
        mPIInfo.buffer
    );
    cudaDeviceSynchronize();

    host_sendParticlesSpeciesDownToUp = sendParticlesSpeciesDownToUp;
    host_sendParticlesSpeciesUpToDown = sendParticlesSpeciesUpToDown;
    countForSendSpeciesDownToUp = device_countForSendSpeciesDownToUp[0];
    countForSendSpeciesUpToDown = device_countForSendSpeciesUpToDown[0];

    sendrecv_particle_y(
        host_sendParticlesSpeciesDownToUp, 
        host_sendParticlesSpeciesUpToDown, 
        host_recvParticlesSpeciesDownToUp, 
        host_recvParticlesSpeciesUpToDown, 
        countForSendSpeciesDownToUp, 
        countForSendSpeciesUpToDown, 
        countForRecvSpeciesDownToUp, 
        countForRecvSpeciesUpToDown,
        mPIInfo
    );

    for (unsigned long long i = 0; i < countForRecvSpeciesDownToUp; i++) {
        particlesSpecies[existNumSpecies + i] = host_recvParticlesSpeciesDownToUp[i];
    }
    existNumSpecies += countForRecvSpeciesDownToUp;
    for (unsigned long long i = 0; i < countForRecvSpeciesUpToDown; i++) {
        particlesSpecies[existNumSpecies + i] = host_recvParticlesSpeciesUpToDown[i];
    }
    existNumSpecies += countForRecvSpeciesUpToDown;
}


