#include "boundary.hpp"


Boundary::Boundary(MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo)
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
    thrust::device_vector<Particle> sendParticlesSpeciesLeft(numForSendParticlesSpeciesLeft);
    thrust::device_vector<Particle> sendParticlesSpeciesRight(numForSendParticlesSpeciesRight);
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

    thrust::host_vector<Particle> host_sendParticlesSpeciesLeft(numForSendParticlesSpeciesLeft);
    thrust::host_vector<Particle> host_sendParticlesSpeciesRight(numForSendParticlesSpeciesRight);
    host_sendParticlesSpeciesLeft = sendParticlesSpeciesLeft;
    host_sendParticlesSpeciesRight = sendParticlesSpeciesRight;

    sendrecv_numParticle_x(
        numForSendParticlesSpeciesLeft, 
        numForSendParticlesSpeciesRight, 
        numForRecvParticlesSpeciesLeft, 
        numForRecvParticlesSpeciesRight, 
        mPIInfo
    );

    thrust::host_vector<Particle> host_recvParticlesSpeciesLeft(numForRecvParticlesSpeciesLeft);
    thrust::host_vector<Particle> host_recvParticlesSpeciesRight(numForRecvParticlesSpeciesRight);
    sendrecv_particle_x(
        host_sendParticlesSpeciesLeft, 
        host_sendParticlesSpeciesRight,  
        host_recvParticlesSpeciesLeft, 
        host_recvParticlesSpeciesRight,  
        mPIInfo
    );

    for (unsigned int i = 0; i < numForRecvParticlesSpeciesLeft; i++) {
        particlesSpecies[existNumSpecies + i] = host_recvParticlesSpeciesLeft[i];
    }
    existNumSpecies += numForRecvParticlesSpeciesLeft;
    for (unsigned int i = 0; i < numForRecvParticlesSpeciesRight; i++) {
        particlesSpecies[existNumSpecies + i] = host_recvParticlesSpeciesRight[i];
    }
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
    thrust::device_vector<Particle> sendParticlesSpeciesDown(numForSendParticlesSpeciesDown);
    thrust::device_vector<Particle> sendParticlesSpeciesUp(numForSendParticlesSpeciesUp);
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

    thrust::host_vector<Particle> host_sendParticlesSpeciesDown(numForSendParticlesSpeciesDown);
    thrust::host_vector<Particle> host_sendParticlesSpeciesUp(numForSendParticlesSpeciesUp);
    host_sendParticlesSpeciesDown = sendParticlesSpeciesDown;
    host_sendParticlesSpeciesUp = sendParticlesSpeciesUp;

    sendrecv_numParticle_y(
        numForSendParticlesSpeciesDown, 
        numForSendParticlesSpeciesUp, 
        numForRecvParticlesSpeciesDown, 
        numForRecvParticlesSpeciesUp, 
        mPIInfo
    );

    thrust::host_vector<Particle> host_recvParticlesSpeciesDown(numForRecvParticlesSpeciesDown);
    thrust::host_vector<Particle> host_recvParticlesSpeciesUp(numForRecvParticlesSpeciesUp);
    sendrecv_particle_y(
        host_sendParticlesSpeciesDown, 
        host_sendParticlesSpeciesUp,  
        host_recvParticlesSpeciesDown, 
        host_recvParticlesSpeciesUp,  
        mPIInfo
    );

    for (unsigned int i = 0; i < numForRecvParticlesSpeciesDown; i++) {
        particlesSpecies[existNumSpecies + i] = host_recvParticlesSpeciesDown[i];
    }
    existNumSpecies += numForRecvParticlesSpeciesDown;
    for (unsigned int i = 0; i < numForRecvParticlesSpeciesUp; i++) {
        particlesSpecies[existNumSpecies + i] = host_recvParticlesSpeciesUp[i];
    }
    existNumSpecies += numForRecvParticlesSpeciesUp;
}

