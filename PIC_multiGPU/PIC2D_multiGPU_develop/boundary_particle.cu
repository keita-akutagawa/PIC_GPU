#include "boundary.hpp"


Boundary::Boundary(MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo), 
      sendParticlesSpeciesLeft(mPIInfo.mpiBufNumParticles), 
      sendParticlesSpeciesRight(mPIInfo.mpiBufNumParticles), 
      sendParticlesSpeciesDown(mPIInfo.mpiBufNumParticles), 
      sendParticlesSpeciesUp(mPIInfo.mpiBufNumParticles), 
      recvParticlesSpeciesLeft(mPIInfo.mpiBufNumParticles), 
      recvParticlesSpeciesRight(mPIInfo.mpiBufNumParticles), 
      recvParticlesSpeciesDown(mPIInfo.mpiBufNumParticles), 
      recvParticlesSpeciesUp(mPIInfo.mpiBufNumParticles) 

{
}


void Boundary::periodicBoundaryParticle_xy(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron
)
{   
    periodicBoundaryParticleOfOneSpecies_x(
        particlesIon,
        mPIInfo.existNumIonPerProcs, 
        mPIInfo.numForSendParticlesIonLeft, 
        mPIInfo.numForSendParticlesIonRight, 
        mPIInfo.numForRecvParticlesIonLeft, 
        mPIInfo.numForRecvParticlesIonRight
    );
    periodicBoundaryParticleOfOneSpecies_x(
        particlesElectron, 
        mPIInfo.existNumElectronPerProcs,  
        mPIInfo.numForSendParticlesElectronLeft, 
        mPIInfo.numForSendParticlesElectronRight, 
        mPIInfo.numForRecvParticlesElectronLeft, 
        mPIInfo.numForRecvParticlesElectronRight
    );
    MPI_Barrier(MPI_COMM_WORLD);

    modifySendNumParticlesSpecies(
        mPIInfo.numForSendParticlesIonCornerLeftDown, 
        mPIInfo.numForSendParticlesIonCornerRightDown, 
        mPIInfo.numForSendParticlesIonCornerLeftUp, 
        mPIInfo.numForSendParticlesIonCornerRightUp, 
        mPIInfo.numForRecvParticlesIonCornerLeftDown, 
        mPIInfo.numForRecvParticlesIonCornerRightDown, 
        mPIInfo.numForRecvParticlesIonCornerLeftUp, 
        mPIInfo.numForRecvParticlesIonCornerRightUp, 
        mPIInfo.numForSendParticlesIonDown, 
        mPIInfo.numForSendParticlesIonUp
    );
    modifySendNumParticlesSpecies(
        mPIInfo.numForSendParticlesElectronCornerLeftDown, 
        mPIInfo.numForSendParticlesElectronCornerRightDown, 
        mPIInfo.numForSendParticlesElectronCornerLeftUp, 
        mPIInfo.numForSendParticlesElectronCornerRightUp, 
        mPIInfo.numForRecvParticlesElectronCornerLeftDown, 
        mPIInfo.numForRecvParticlesElectronCornerRightDown, 
        mPIInfo.numForRecvParticlesElectronCornerLeftUp, 
        mPIInfo.numForRecvParticlesElectronCornerRightUp, 
        mPIInfo.numForSendParticlesElectronDown, 
        mPIInfo.numForSendParticlesElectronUp
    );
    MPI_Barrier(MPI_COMM_WORLD);

    periodicBoundaryParticleOfOneSpecies_y(
        particlesIon,
        mPIInfo.existNumIonPerProcs, 
        mPIInfo.numForSendParticlesIonDown, 
        mPIInfo.numForSendParticlesIonUp, 
        mPIInfo.numForRecvParticlesIonDown, 
        mPIInfo.numForRecvParticlesIonUp
    );
    periodicBoundaryParticleOfOneSpecies_y(
        particlesElectron, 
        mPIInfo.existNumElectronPerProcs, 
        mPIInfo.numForSendParticlesElectronDown, 
        mPIInfo.numForSendParticlesElectronUp, 
        mPIInfo.numForRecvParticlesElectronDown, 
        mPIInfo.numForRecvParticlesElectronUp
    );
    MPI_Barrier(MPI_COMM_WORLD);
}


__global__ void periodicBoundaryParticle_x_kernel(
    Particle* particlesSpecies, 
    Particle* sendParticlesSpeciesLeft, 
    Particle* sendParticlesSpeciesRight, 
    unsigned int* countForSendParticlesSpeciesLeft, 
    unsigned int* countForSendParticlesSpeciesRight, 
    const unsigned long long existNumSpecies, 
    const float xminForProcs, const float xmaxForProcs, 
    const int buffer
)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (particlesSpecies[i].isMPISendRight) {
            unsigned int particleIndex = atomicAdd(&(countForSendParticlesSpeciesRight[0]), 1);
            particlesSpecies[i].isMPISendRight = false;
            Particle sendParticle = particlesSpecies[i];
            if (sendParticle.x > device_xmax - buffer * device_dx + device_EPS) {
                sendParticle.x = sendParticle.x - device_xmax + device_EPS;
            }
            sendParticlesSpeciesRight[particleIndex] = sendParticle;
        }

        if (particlesSpecies[i].isMPISendLeft) {
            unsigned int particleIndex = atomicAdd(&(countForSendParticlesSpeciesLeft[0]), 1);
            particlesSpecies[i].isMPISendLeft = false;
            Particle sendParticle = particlesSpecies[i];
            if (sendParticle.x < device_xmin + buffer * device_dx - device_EPS) {
                sendParticle.x = sendParticle.x + device_xmax - device_EPS;
            }
            sendParticlesSpeciesLeft[particleIndex] = sendParticle;
        }
    }
}

void Boundary::periodicBoundaryParticleOfOneSpecies_x(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies, 
    unsigned int& numForSendParticlesSpeciesLeft, 
    unsigned int& numForSendParticlesSpeciesRight, 
    unsigned int& numForRecvParticlesSpeciesLeft, 
    unsigned int& numForRecvParticlesSpeciesRight
)
{
    thrust::device_vector<unsigned int> countForSendParticlesSpeciesLeft(1, 0); 
    thrust::device_vector<unsigned int> countForSendParticlesSpeciesRight(1, 0); 

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    periodicBoundaryParticle_x_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesLeft.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesRight.data()), 
        thrust::raw_pointer_cast(countForSendParticlesSpeciesLeft.data()), 
        thrust::raw_pointer_cast(countForSendParticlesSpeciesRight.data()), 
        existNumSpecies, 
        mPIInfo.xminForProcs, mPIInfo.xmaxForProcs, 
        mPIInfo.buffer
    );
    cudaDeviceSynchronize();

    sendrecv_numParticle_x(
        numForSendParticlesSpeciesLeft, 
        numForSendParticlesSpeciesRight, 
        numForRecvParticlesSpeciesLeft, 
        numForRecvParticlesSpeciesRight, 
        mPIInfo
    );

    sendrecv_particle_x(
        sendParticlesSpeciesLeft, 
        sendParticlesSpeciesRight,  
        recvParticlesSpeciesLeft, 
        recvParticlesSpeciesRight,  
        numForSendParticlesSpeciesLeft, 
        numForSendParticlesSpeciesRight, 
        numForRecvParticlesSpeciesLeft, 
        numForRecvParticlesSpeciesRight, 
        mPIInfo
    );

    thrust::copy(
        recvParticlesSpeciesLeft.begin(), 
        recvParticlesSpeciesLeft.begin() + numForRecvParticlesSpeciesLeft,
        particlesSpecies.begin() + existNumSpecies
    );
    existNumSpecies += numForRecvParticlesSpeciesLeft;
    thrust::copy(
        recvParticlesSpeciesRight.begin(), 
        recvParticlesSpeciesRight.begin() + numForRecvParticlesSpeciesRight,
        particlesSpecies.begin() + existNumSpecies
    );
    existNumSpecies += numForRecvParticlesSpeciesRight;
}


void Boundary::modifySendNumParticlesSpecies(
    const unsigned int& numForSendParticlesSpeciesCornerLeftDown, 
    const unsigned int& numForSendParticlesSpeciesCornerRightDown, 
    const unsigned int& numForSendParticlesSpeciesCornerLeftUp, 
    const unsigned int& numForSendParticlesSpeciesCornerRightUp, 
    unsigned int& numForRecvParticlesSpeciesCornerLeftDown, 
    unsigned int& numForRecvParticlesSpeciesCornerRightDown, 
    unsigned int& numForRecvParticlesSpeciesCornerLeftUp, 
    unsigned int& numForRecvParticlesSpeciesCornerRightUp, 
    unsigned int& numForSendParticlesSpeciesDown, 
    unsigned int& numForSendParticlesSpeciesUp
)
{
    sendrecv_numParticle_corner(
        numForSendParticlesSpeciesCornerLeftDown, 
        numForSendParticlesSpeciesCornerRightDown, 
        numForSendParticlesSpeciesCornerLeftUp, 
        numForSendParticlesSpeciesCornerRightUp, 
        numForRecvParticlesSpeciesCornerLeftDown, 
        numForRecvParticlesSpeciesCornerRightDown, 
        numForRecvParticlesSpeciesCornerLeftUp, 
        numForRecvParticlesSpeciesCornerRightUp, 
        mPIInfo
    );

    numForSendParticlesSpeciesDown += numForRecvParticlesSpeciesCornerLeftDown
                                    + numForRecvParticlesSpeciesCornerRightDown;
    numForSendParticlesSpeciesUp   += numForRecvParticlesSpeciesCornerLeftUp
                                    + numForRecvParticlesSpeciesCornerRightUp;

}



__global__ void periodicBoundaryParticle_y_kernel(
    Particle* particlesSpecies, 
    Particle* sendParticlesSpeciesDown, 
    Particle* sendParticlesSpeciesUp, 
    unsigned int* countForSendParticlesSpeciesDown, 
    unsigned int* countForSendParticlesSpeciesUp, 
    const unsigned long long existNumSpecies, 
    const float yminForProcs, const float ymaxForProcs, 
    const int buffer
)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (particlesSpecies[i].isMPISendUp) {
            unsigned int particleIndex = atomicAdd(&(countForSendParticlesSpeciesUp[0]), 1);
            particlesSpecies[i].isMPISendUp = false;
            Particle sendParticle = particlesSpecies[i];
            if (sendParticle.y > device_ymax - buffer * device_dy + device_EPS) {
                sendParticle.y = sendParticle.y - device_ymax + device_EPS;
            }
            sendParticlesSpeciesUp[particleIndex] = sendParticle;
        }

        if (particlesSpecies[i].isMPISendDown) {
            unsigned int particleIndex = atomicAdd(&(countForSendParticlesSpeciesDown[0]), 1);
            particlesSpecies[i].isMPISendDown = false;
            Particle sendParticle = particlesSpecies[i];
            if (sendParticle.y < device_ymin + buffer * device_dy - device_EPS) {
                sendParticle.y = sendParticle.y + device_ymax - device_EPS;
            }
            sendParticlesSpeciesDown[particleIndex] = sendParticle;
        }
    }
}

void Boundary::periodicBoundaryParticleOfOneSpecies_y(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies, 
    unsigned int& numForSendParticlesSpeciesDown, 
    unsigned int& numForSendParticlesSpeciesUp, 
    unsigned int& numForRecvParticlesSpeciesDown, 
    unsigned int& numForRecvParticlesSpeciesUp
)
{
    thrust::device_vector<unsigned int> countForSendParticlesSpeciesDown(1, 0); 
    thrust::device_vector<unsigned int> countForSendParticlesSpeciesUp(1, 0); 

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    periodicBoundaryParticle_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesDown.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesUp.data()), 
        thrust::raw_pointer_cast(countForSendParticlesSpeciesDown.data()), 
        thrust::raw_pointer_cast(countForSendParticlesSpeciesUp.data()), 
        existNumSpecies, 
        mPIInfo.yminForProcs, mPIInfo.ymaxForProcs, 
        mPIInfo.buffer
    );
    cudaDeviceSynchronize();

    sendrecv_numParticle_y(
        numForSendParticlesSpeciesDown, 
        numForSendParticlesSpeciesUp, 
        numForRecvParticlesSpeciesDown, 
        numForRecvParticlesSpeciesUp, 
        mPIInfo
    );

    sendrecv_particle_y(
        sendParticlesSpeciesDown, 
        sendParticlesSpeciesUp,  
        recvParticlesSpeciesDown, 
        recvParticlesSpeciesUp,  
        numForSendParticlesSpeciesDown, 
        numForSendParticlesSpeciesUp, 
        numForRecvParticlesSpeciesDown, 
        numForRecvParticlesSpeciesUp, 
        mPIInfo
    );

    thrust::copy(
        recvParticlesSpeciesDown.begin(), 
        recvParticlesSpeciesDown.begin() + numForRecvParticlesSpeciesDown,
        particlesSpecies.begin() + existNumSpecies
    );
    existNumSpecies += numForRecvParticlesSpeciesDown;
    thrust::copy(
        recvParticlesSpeciesUp.begin(), 
        recvParticlesSpeciesUp.begin() + numForRecvParticlesSpeciesUp,
        particlesSpecies.begin() + existNumSpecies
    );
    existNumSpecies += numForRecvParticlesSpeciesUp;
}

