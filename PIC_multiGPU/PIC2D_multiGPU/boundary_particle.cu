#include "boundary.hpp"


Boundary::Boundary(MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo)
{
}


struct IsExistTransform
{
    __host__ __device__
    unsigned int operator()(const Particle& p) const {
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
        mPIInfo.numForSendParticlesIonLeftward, 
        mPIInfo.numForSendParticlesIonRightward, 
        mPIInfo.numForRecvParticlesIonLeftward, 
        mPIInfo.numForRecvParticlesIonRightward
    );
    periodicBoundaryParticleOfOneSpeciesX(
        particlesElectron, 
        mPIInfo.existNumElectronPerProcs,  
        mPIInfo.numForSendParticlesElectronLeftward, 
        mPIInfo.numForSendParticlesElectronRightward, 
        mPIInfo.numForRecvParticlesElectronLeftward, 
        mPIInfo.numForRecvParticlesElectronRightward
    );
    MPI_Barrier(MPI_COMM_WORLD);

    periodicBoundaryParticleOfOneSpeciesY(
        particlesIon,
        mPIInfo.existNumIonPerProcs, 
        mPIInfo.numForSendParticlesIonDownward, 
        mPIInfo.numForSendParticlesIonUpward, 
        mPIInfo.numForRecvParticlesIonDownward, 
        mPIInfo.numForRecvParticlesIonUpward
    );
    periodicBoundaryParticleOfOneSpeciesY(
        particlesElectron, 
        mPIInfo.existNumElectronPerProcs, 
        mPIInfo.numForSendParticlesElectronDownward, 
        mPIInfo.numForSendParticlesElectronUpward, 
        mPIInfo.numForRecvParticlesElectronDownward, 
        mPIInfo.numForRecvParticlesElectronUpward
    );
    MPI_Barrier(MPI_COMM_WORLD);
}


__global__ void periodicBoundaryParticleX_kernel(
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
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (particlesSpecies[i].x - (xminForProcs - buffer * device_dx) < device_EPS) {
            particlesSpecies[i].isExist = false;
        }

        if (particlesSpecies[i].x - (xmaxForProcs + buffer * device_dx) > -device_EPS) {
            particlesSpecies[i].isExist = false;
        }

        if (particlesSpecies[i].isMPISendRightward) {
            unsigned int particleIndex = atomicAdd(&(countForSendParticlesSpeciesRightward[0]), 1);
            particlesSpecies[i].isMPISendRightward = false;
            Particle sendParticle = particlesSpecies[i];
            if (sendParticle.x > device_xmax - buffer * device_dx) {
                sendParticle.x -= device_xmax + device_EPS;
            }
            sendParticlesSpeciesRightward[particleIndex] = sendParticle;
        }

        if (particlesSpecies[i].isMPISendLeftward) {
            unsigned int particleIndex = atomicAdd(&(countForSendParticlesSpeciesLeftward[0]), 1);
            particlesSpecies[i].isMPISendLeftward = false;
            Particle sendParticle = particlesSpecies[i];
            if (sendParticle.x < device_xmin + buffer * device_dx) {
                sendParticle.x += device_xmax - device_EPS;
            }
            sendParticlesSpeciesLeftward[particleIndex] = sendParticle;
        }
    }
}

void Boundary::periodicBoundaryParticleOfOneSpeciesX(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies, 
    unsigned int& numForSendParticlesSpeciesLeftward, 
    unsigned int& numForSendParticlesSpeciesRightward, 
    unsigned int& numForRecvParticlesSpeciesLeftward, 
    unsigned int& numForRecvParticlesSpeciesRightward
)
{
    thrust::device_vector<Particle> sendParticlesSpeciesLeftward(numForSendParticlesSpeciesLeftward);
    thrust::device_vector<Particle> sendParticlesSpeciesRightward(numForSendParticlesSpeciesRightward);
    thrust::device_vector<unsigned int> countForSendParticlesSpeciesLeftward(1, 0); 
    thrust::device_vector<unsigned int> countForSendParticlesSpeciesRightward(1, 0); 

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    periodicBoundaryParticleX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
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

    thrust::host_vector<Particle> host_sendParticlesSpeciesLeftward(numForSendParticlesSpeciesLeftward);
    thrust::host_vector<Particle> host_sendParticlesSpeciesRightward(numForSendParticlesSpeciesRightward);
    host_sendParticlesSpeciesLeftward = sendParticlesSpeciesLeftward;
    host_sendParticlesSpeciesRightward = sendParticlesSpeciesRightward;
    numForSendParticlesSpeciesLeftward = countForSendParticlesSpeciesLeftward[0];
    numForSendParticlesSpeciesRightward = countForSendParticlesSpeciesRightward[0];

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


__global__ void periodicBoundaryParticleY_kernel(
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
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (particlesSpecies[i].y - (yminForProcs - buffer * device_dy) < device_EPS) {
            particlesSpecies[i].isExist = false;
        }

        if (particlesSpecies[i].y - (ymaxForProcs + buffer * device_dy) > -device_EPS) {
            particlesSpecies[i].isExist = false;
        }

        if (particlesSpecies[i].isMPISendUpward) {
            unsigned int particleIndex = atomicAdd(&(countForSendParticlesSpeciesUpward[0]), 1);
            particlesSpecies[i].isMPISendUpward = false;
            Particle sendParticle = particlesSpecies[i];
            if (sendParticle.y > device_ymax - buffer * device_dy) {
                sendParticle.y -= device_ymax + device_EPS;
            }
            sendParticlesSpeciesUpward[particleIndex] = sendParticle;
        }

        if (particlesSpecies[i].isMPISendDownward) {
            unsigned int particleIndex = atomicAdd(&(countForSendParticlesSpeciesDownward[0]), 1);
            particlesSpecies[i].isMPISendDownward = false;
            Particle sendParticle = particlesSpecies[i];
            if (sendParticle.y < device_ymin + buffer * device_dy) {
                sendParticle.y += device_ymax - device_EPS;
            }
            sendParticlesSpeciesDownward[particleIndex] = sendParticle;
        }
    }
}

void Boundary::periodicBoundaryParticleOfOneSpeciesY(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies, 
    unsigned int& numForSendParticlesSpeciesDownward, 
    unsigned int& numForSendParticlesSpeciesUpward, 
    unsigned int& numForRecvParticlesSpeciesDownward, 
    unsigned int& numForRecvParticlesSpeciesUpward
)
{
    thrust::device_vector<Particle> sendParticlesSpeciesDownward(numForSendParticlesSpeciesDownward);
    thrust::device_vector<Particle> sendParticlesSpeciesUpward(numForSendParticlesSpeciesUpward);
    thrust::device_vector<unsigned int> countForSendParticlesSpeciesDownward(1, 0); 
    thrust::device_vector<unsigned int> countForSendParticlesSpeciesUpward(1, 0); 

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    periodicBoundaryParticleY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
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

    thrust::host_vector<Particle> host_sendParticlesSpeciesDownward(numForSendParticlesSpeciesDownward);
    thrust::host_vector<Particle> host_sendParticlesSpeciesUpward(numForSendParticlesSpeciesUpward);
    host_sendParticlesSpeciesDownward = sendParticlesSpeciesDownward;
    host_sendParticlesSpeciesUpward = sendParticlesSpeciesUpward;
    numForSendParticlesSpeciesDownward = countForSendParticlesSpeciesDownward[0];
    numForSendParticlesSpeciesUpward = countForSendParticlesSpeciesUpward[0];

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

