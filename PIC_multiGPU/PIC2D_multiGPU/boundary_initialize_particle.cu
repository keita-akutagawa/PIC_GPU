#include "boundary.hpp"


void Boundary::boundaryForInitializeParticle(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron
)
{
    boundaryForInitializeParticleOfOneSpeciesX(
        particlesIon, 
        mPIInfo.existNumIonPerProcs, 
        mPIInfo.numForSendParticlesIonLeftward, 
        mPIInfo.numForSendParticlesIonRightward, 
        mPIInfo.numForRecvParticlesIonLeftward, 
        mPIInfo.numForRecvParticlesIonRightward
    ); 
    boundaryForInitializeParticleOfOneSpeciesX(
        particlesElectron, 
        mPIInfo.existNumElectronPerProcs,  
        mPIInfo.numForSendParticlesElectronLeftward, 
        mPIInfo.numForSendParticlesElectronRightward, 
        mPIInfo.numForRecvParticlesElectronLeftward, 
        mPIInfo.numForRecvParticlesElectronRightward
    ); 

    MPI_Barrier(MPI_COMM_WORLD);

    boundaryForInitializeParticleOfOneSpeciesY(
        particlesIon, 
        mPIInfo.existNumIonPerProcs, 
        mPIInfo.numForSendParticlesIonDownward, 
        mPIInfo.numForSendParticlesIonUpward, 
        mPIInfo.numForRecvParticlesIonDownward, 
        mPIInfo.numForRecvParticlesIonUpward
    ); 
    boundaryForInitializeParticleOfOneSpeciesY(
        particlesElectron, 
        mPIInfo.existNumElectronPerProcs,  
        mPIInfo.numForSendParticlesElectronDownward, 
        mPIInfo.numForSendParticlesElectronUpward, 
        mPIInfo.numForRecvParticlesElectronDownward, 
        mPIInfo.numForRecvParticlesElectronUpward
    ); 

    MPI_Barrier(MPI_COMM_WORLD);
}


__global__ void boundaryForInitializeX_count_kernel(
    Particle* particlesSpecies, 
    unsigned int* countForSendParticlesSpeciesLeftward, 
    unsigned int* countForSendParticlesSpeciesRightward, 
    const unsigned long long existNumSpecies, 
    const float xminForProcs, const float xmaxForProcs, 
    const int buffer
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (xmaxForProcs - buffer * device_dx < particlesSpecies[i].x && particlesSpecies[i].x < xmaxForProcs) {
            atomicAdd(&(countForSendParticlesSpeciesRightward[0]), 1);
        }

        if (xminForProcs < particlesSpecies[i].x && particlesSpecies[i].x < xminForProcs + buffer * device_dx) {
            atomicAdd(&(countForSendParticlesSpeciesLeftward[0]), 1);
        }
    }
}


__global__ void boundaryForInitializeX_kernel(
    Particle* particlesSpecies, 
    Particle* sendParticlesSpeciesLeftward, 
    Particle* sendParticlesSpeciesRightward, 
    unsigned int* countForSendParticlesSpeciesLeftward, 
    unsigned int* countForSendParticlesSpeciesRightward, 
    const unsigned long long existNumSpecies, 
    const float xminForProcs, const float xmaxForProcs, 
    const int buffer
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (xmaxForProcs - buffer * device_dx < particlesSpecies[i].x && particlesSpecies[i].x < xmaxForProcs) {
            unsigned long long particleIndex = atomicAdd(&(countForSendParticlesSpeciesRightward[0]), 1);
            Particle sendParticle = particlesSpecies[i];
            if (device_xmax - buffer * device_dx < sendParticle.x) {
                sendParticle.x -= device_xmax + device_EPS;
            }
            sendParticlesSpeciesRightward[particleIndex] = sendParticle;
        }

        if (xminForProcs < particlesSpecies[i].x && particlesSpecies[i].x < xminForProcs + buffer * device_dx) {
            unsigned long long particleIndex = atomicAdd(&(countForSendParticlesSpeciesLeftward[0]), 1);
            Particle sendParticle = particlesSpecies[i];
            if (sendParticle.x < device_xmin + buffer * device_dx) {
                sendParticle.x += device_xmax - device_EPS;
            }
            sendParticlesSpeciesLeftward[particleIndex] = sendParticle;
        }
    }
}

void Boundary::boundaryForInitializeParticleOfOneSpeciesX(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies, 
    unsigned int& numForSendParticlesSpeciesLeftward, 
    unsigned int& numForSendParticlesSpeciesRightward, 
    unsigned int& numForRecvParticlesSpeciesLeftward, 
    unsigned int& numForRecvParticlesSpeciesRightward
)
{
    thrust::device_vector<unsigned int> countForSendParticlesSpeciesLeftward(1, 0); 
    thrust::device_vector<unsigned int> countForSendParticlesSpeciesRightward(1, 0); 

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    boundaryForInitializeX_count_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(countForSendParticlesSpeciesLeftward.data()), 
        thrust::raw_pointer_cast(countForSendParticlesSpeciesRightward.data()), 
        existNumSpecies, 
        mPIInfo.xminForProcs, mPIInfo.xmaxForProcs, 
        mPIInfo.buffer
    );
    cudaDeviceSynchronize();

    numForSendParticlesSpeciesLeftward = countForSendParticlesSpeciesLeftward[0];
    numForSendParticlesSpeciesRightward = countForSendParticlesSpeciesRightward[0];

    thrust::device_vector<Particle> sendParticlesSpeciesLeftward(numForSendParticlesSpeciesLeftward);
    thrust::device_vector<Particle> sendParticlesSpeciesRightward(numForSendParticlesSpeciesRightward);
    countForSendParticlesSpeciesLeftward[0] = 0;
    countForSendParticlesSpeciesRightward[0] = 0;

    boundaryForInitializeX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesLeftward.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesRightward.data()), 
        thrust::raw_pointer_cast(countForSendParticlesSpeciesLeftward.data()), 
        thrust::raw_pointer_cast(countForSendParticlesSpeciesRightward.data()), 
        existNumSpecies, 
        mPIInfo.xminForProcs, mPIInfo.xmaxForProcs, 
        mPIInfo.buffer
    );
    cudaDeviceSynchronize();

    thrust::host_vector<Particle> host_sendParticlesSpeciesLeftward(numForSendParticlesSpeciesLeftward);
    thrust::host_vector<Particle> host_sendParticlesSpeciesRightward(numForSendParticlesSpeciesRightward);
    host_sendParticlesSpeciesLeftward = sendParticlesSpeciesLeftward;
    host_sendParticlesSpeciesRightward = sendParticlesSpeciesRightward;

    sendrecv_num_particle_x(
        numForSendParticlesSpeciesLeftward, 
        numForSendParticlesSpeciesRightward, 
        numForRecvParticlesSpeciesLeftward, 
        numForRecvParticlesSpeciesRightward, 
        mPIInfo
    );

    thrust::host_vector<Particle> host_recvParticlesSpeciesLeftward(numForRecvParticlesSpeciesLeftward);
    thrust::host_vector<Particle> host_recvParticlesSpeciesRightward(numForRecvParticlesSpeciesRightward);

    sendrecv_particle_x(
        host_sendParticlesSpeciesLeftward, 
        host_sendParticlesSpeciesRightward,  
        host_recvParticlesSpeciesLeftward, 
        host_recvParticlesSpeciesRightward,  
        mPIInfo
    );

    for (unsigned int i = 0; i < numForRecvParticlesSpeciesLeftward; i++) {
        particlesSpecies[existNumSpecies + i] = host_recvParticlesSpeciesLeftward[i];
    }
    existNumSpecies += numForRecvParticlesSpeciesLeftward;
    for (unsigned int i = 0; i < numForRecvParticlesSpeciesRightward; i++) {
        particlesSpecies[existNumSpecies + i] = host_recvParticlesSpeciesRightward[i];
    }
    existNumSpecies += numForRecvParticlesSpeciesRightward;
}


__global__ void boundaryForInitializeY_count_kernel(
    Particle* particlesSpecies, 
    unsigned int* countForSendParticlesSpeciesDownward, 
    unsigned int* countForSendParticlesSpeciesUpward, 
    const unsigned long long existNumSpecies, 
    const float yminForProcs, const float ymaxForProcs, 
    const int buffer
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (ymaxForProcs - buffer * device_dy < particlesSpecies[i].y && particlesSpecies[i].y < ymaxForProcs) {
            atomicAdd(&(countForSendParticlesSpeciesUpward[0]), 1);
        }

        if (yminForProcs < particlesSpecies[i].y && particlesSpecies[i].y < yminForProcs + buffer * device_dy) {
            atomicAdd(&(countForSendParticlesSpeciesDownward[0]), 1);
        }
    }
}


__global__ void boundaryForInitializeY_kernel(
    Particle* particlesSpecies, 
    Particle* sendParticlesSpeciesDownward, 
    Particle* sendParticlesSpeciesUpward, 
    unsigned int* countForSendParticlesSpeciesDownward, 
    unsigned int* countForSendParticlesSpeciesUpward, 
    const unsigned long long existNumSpecies, 
    const float yminForProcs, const float ymaxForProcs, 
    const int buffer
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (ymaxForProcs - buffer * device_dy < particlesSpecies[i].y && particlesSpecies[i].y < ymaxForProcs) {
            unsigned long long particleIndex = atomicAdd(&(countForSendParticlesSpeciesUpward[0]), 1);
            Particle sendParticle = particlesSpecies[i];
            if (device_ymax - buffer * device_dy < sendParticle.y) {
                sendParticle.y -= device_ymax + device_EPS;
            }
            sendParticlesSpeciesUpward[particleIndex] = sendParticle;
        }

        if (yminForProcs < particlesSpecies[i].y && particlesSpecies[i].y < yminForProcs + buffer * device_dy) {
            unsigned long long particleIndex = atomicAdd(&(countForSendParticlesSpeciesDownward[0]), 1);
            Particle sendParticle = particlesSpecies[i];
            if (sendParticle.y < device_ymin + buffer * device_dy) {
                sendParticle.y += device_ymax - device_EPS;
            }
            sendParticlesSpeciesDownward[particleIndex] = sendParticle;
        }
    }
}

void Boundary::boundaryForInitializeParticleOfOneSpeciesY(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies, 
    unsigned int& numForSendParticlesSpeciesDownward, 
    unsigned int& numForSendParticlesSpeciesUpward, 
    unsigned int& numForRecvParticlesSpeciesDownward, 
    unsigned int& numForRecvParticlesSpeciesUpward
)
{
    thrust::device_vector<unsigned int> countForSendParticlesSpeciesDownward(1, 0); 
    thrust::device_vector<unsigned int> countForSendParticlesSpeciesUpward(1, 0); 

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    boundaryForInitializeY_count_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(countForSendParticlesSpeciesDownward.data()), 
        thrust::raw_pointer_cast(countForSendParticlesSpeciesUpward.data()), 
        existNumSpecies, 
        mPIInfo.yminForProcs, mPIInfo.ymaxForProcs, 
        mPIInfo.buffer
    );
    cudaDeviceSynchronize();

    numForSendParticlesSpeciesDownward = countForSendParticlesSpeciesDownward[0];
    numForSendParticlesSpeciesUpward = countForSendParticlesSpeciesUpward[0];

    thrust::device_vector<Particle> sendParticlesSpeciesDownward(numForSendParticlesSpeciesDownward);
    thrust::device_vector<Particle> sendParticlesSpeciesUpward(numForSendParticlesSpeciesUpward);
    countForSendParticlesSpeciesDownward[0] = 0;
    countForSendParticlesSpeciesUpward[0] = 0;

    boundaryForInitializeY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesDownward.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesUpward.data()), 
        thrust::raw_pointer_cast(countForSendParticlesSpeciesDownward.data()), 
        thrust::raw_pointer_cast(countForSendParticlesSpeciesUpward.data()), 
        existNumSpecies, 
        mPIInfo.yminForProcs, mPIInfo.ymaxForProcs, 
        mPIInfo.buffer
    );
    cudaDeviceSynchronize();

    thrust::host_vector<Particle> host_sendParticlesSpeciesDownward(numForSendParticlesSpeciesDownward);
    thrust::host_vector<Particle> host_sendParticlesSpeciesUpward(numForSendParticlesSpeciesUpward);
    host_sendParticlesSpeciesDownward = sendParticlesSpeciesDownward;
    host_sendParticlesSpeciesUpward = sendParticlesSpeciesUpward;

    sendrecv_num_particle_y(
        numForSendParticlesSpeciesDownward, 
        numForSendParticlesSpeciesUpward, 
        numForRecvParticlesSpeciesDownward, 
        numForRecvParticlesSpeciesUpward, 
        mPIInfo
    );

    thrust::host_vector<Particle> host_recvParticlesSpeciesDownward(numForRecvParticlesSpeciesDownward);
    thrust::host_vector<Particle> host_recvParticlesSpeciesUpward(numForRecvParticlesSpeciesUpward);
    sendrecv_particle_y(
        host_sendParticlesSpeciesDownward, 
        host_sendParticlesSpeciesUpward,  
        host_recvParticlesSpeciesDownward, 
        host_recvParticlesSpeciesUpward,  
        mPIInfo
    );

    for (unsigned int i = 0; i < numForRecvParticlesSpeciesDownward; i++) {
        particlesSpecies[existNumSpecies + i] = host_recvParticlesSpeciesDownward[i];
    }
    existNumSpecies += numForRecvParticlesSpeciesDownward;
    for (unsigned int i = 0; i < numForRecvParticlesSpeciesUpward; i++) {
        particlesSpecies[existNumSpecies + i] = host_recvParticlesSpeciesUpward[i];
    }
    existNumSpecies += numForRecvParticlesSpeciesUpward;
}


