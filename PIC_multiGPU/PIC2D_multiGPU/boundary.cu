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

      device_countForSendSpeciesLeftToRight(1, 0), 
      device_countForSendSpeciesRightToLeft(1, 0), 
      countForSendSpeciesLeftToRight(0),
      countForSendSpeciesRightToLeft(0),
      countForRecvSpeciesLeftToRight(0),
      countForRecvSpeciesRightToLeft(0),

      device_countForSendSpeciesUpToDown(1, 0), 
      device_countForSendSpeciesDownToUp(1, 0), 
      countForSendSpeciesUpToDown(0),
      countForSendSpeciesDownToUp(0),
      countForRecvSpeciesUpToDown(0),
      countForRecvSpeciesDownToUp(0)
{
}


struct IsExistTransform
{
    __host__ __device__
    unsigned long long operator()(const Particle& p) const {
        return p.isExist ? 1 : 0;
    }
};


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

__global__ void boundaryForInitializeY_kernel(
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
        if (yminForProcs < particlesSpecies[i].y && particlesSpecies[i].y < yminForProcs + buffer * device_dy) {
            unsigned long long particleIndex = atomicAdd(&(countForSendSpeciesUpToDown[0]), 1);
            Particle sendParticle = particlesSpecies[i];
            if (sendParticle.y < device_ymin + buffer * device_dy) {
                sendParticle.y += device_ymax;
            }
            sendParticlesSpeciesUpToDown[particleIndex] = sendParticle;
        }

        if (ymaxForProcs - buffer * device_dy < particlesSpecies[i].y && particlesSpecies[i].y < ymaxForProcs) {
            unsigned long long particleIndex = atomicAdd(&(countForSendSpeciesDownToUp[0]), 1);
            Particle sendParticle = particlesSpecies[i];
            if (device_ymax - buffer * device_dy < sendParticle.y) {
                sendParticle.y -= device_ymax;
            }
            sendParticlesSpeciesDownToUp[particleIndex] = sendParticle;
        }
    }
}

void Boundary::boundaryForInitialize(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies
)
{
    device_countForSendSpeciesLeftToRight[0] = 0; 
    device_countForSendSpeciesRightToLeft[0] = 0; 
    countForSendSpeciesLeftToRight = 0;
    countForSendSpeciesRightToLeft = 0;
    countForRecvSpeciesLeftToRight = 0;
    countForRecvSpeciesRightToLeft = 0;
    device_countForSendSpeciesUpToDown[0] = 0; 
    device_countForSendSpeciesDownToUp[0] = 0; 
    countForSendSpeciesUpToDown = 0;
    countForSendSpeciesDownToUp = 0;
    countForRecvSpeciesUpToDown = 0;
    countForRecvSpeciesDownToUp = 0;

    float xminForProcs = xmin + (xmax - xmin) / mPIInfo.gridX * mPIInfo.localGridX;
    float xmaxForProcs = xmin + (xmax - xmin) / mPIInfo.gridX * (mPIInfo.localGridX + 1);
    float yminForProcs = ymin + (ymax - ymin) / mPIInfo.gridY * mPIInfo.localGridY;
    float ymaxForProcs = ymin + (ymax - ymin) / mPIInfo.gridY * (mPIInfo.localGridY + 1);


    dim3 threadsPerBlockForX(256);
    dim3 blocksPerGridForX((existNumSpecies + threadsPerBlockForX.x - 1) / threadsPerBlockForX.x);

    boundaryForInitializeX_kernel<<<blocksPerGridForX, threadsPerBlockForX>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesLeftToRight.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesRightToLeft.data()), 
        thrust::raw_pointer_cast(device_countForSendSpeciesLeftToRight.data()), 
        thrust::raw_pointer_cast(device_countForSendSpeciesRightToLeft.data()), 
        existNumSpecies, 
        xminForProcs, xmaxForProcs, 
        mPIInfo.buffer
    );
    cudaDeviceSynchronize();

    countForSendSpeciesLeftToRight = device_countForSendSpeciesLeftToRight[0];
    countForSendSpeciesRightToLeft = device_countForSendSpeciesRightToLeft[0];

    sendrecv_particle_x(
        sendParticlesSpeciesLeftToRight, 
        sendParticlesSpeciesRightToLeft, 
        recvParticlesSpeciesLeftToRight, 
        recvParticlesSpeciesRightToLeft, 
        countForSendSpeciesLeftToRight, 
        countForSendSpeciesRightToLeft, 
        countForRecvSpeciesLeftToRight, 
        countForRecvSpeciesRightToLeft, 
        mPIInfo
    );

    for (unsigned long long i = 0; i < countForRecvSpeciesLeftToRight; i++) {
        particlesSpecies[existNumSpecies + i] = recvParticlesSpeciesLeftToRight[i];
    }
    existNumSpecies += countForRecvSpeciesLeftToRight;
    for (unsigned long long i = 0; i < countForRecvSpeciesRightToLeft; i++) {
        particlesSpecies[existNumSpecies + i] = recvParticlesSpeciesRightToLeft[i];
    }
    existNumSpecies += countForRecvSpeciesRightToLeft;

    MPI_Barrier(MPI_COMM_WORLD);


    dim3 threadsPerBlockForY(256);
    dim3 blocksPerGridForY((existNumSpecies + threadsPerBlockForY.x - 1) / threadsPerBlockForY.x);

    boundaryForInitializeY_kernel<<<blocksPerGridForY, threadsPerBlockForY>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesUpToDown.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesDownToUp.data()), 
        thrust::raw_pointer_cast(device_countForSendSpeciesUpToDown.data()), 
        thrust::raw_pointer_cast(device_countForSendSpeciesDownToUp.data()), 
        existNumSpecies, 
        yminForProcs, ymaxForProcs, 
        mPIInfo.buffer
    );
    cudaDeviceSynchronize();

    countForSendSpeciesUpToDown = device_countForSendSpeciesUpToDown[0];
    countForSendSpeciesDownToUp = device_countForSendSpeciesDownToUp[0];

    sendrecv_particle_y(
        sendParticlesSpeciesUpToDown, 
        sendParticlesSpeciesDownToUp, 
        recvParticlesSpeciesUpToDown, 
        recvParticlesSpeciesDownToUp, 
        countForSendSpeciesUpToDown, 
        countForSendSpeciesDownToUp, 
        countForRecvSpeciesUpToDown, 
        countForRecvSpeciesDownToUp, 
        mPIInfo
    );

    for (unsigned long long i = 0; i < countForRecvSpeciesUpToDown; i++) {
        particlesSpecies[existNumSpecies + i] = recvParticlesSpeciesUpToDown[i];
    }
    existNumSpecies += countForRecvSpeciesUpToDown;
    for (unsigned long long i = 0; i < countForRecvSpeciesDownToUp; i++) {
        particlesSpecies[existNumSpecies + i] = recvParticlesSpeciesDownToUp[i];
    }
    existNumSpecies += countForRecvSpeciesDownToUp;
}


void Boundary::periodicBoundaryParticleXY(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron
)
{
    MPI_Barrier(MPI_COMM_WORLD);

    periodicBoundaryParticleOfOneSpeciesX(particlesIon, existNumIonPerProcs);
    periodicBoundaryParticleOfOneSpeciesX(particlesElectron, existNumElectronPerProcs);

    MPI_Barrier(MPI_COMM_WORLD);

    periodicBoundaryParticleOfOneSpeciesY(particlesIon, existNumIonPerProcs);
    periodicBoundaryParticleOfOneSpeciesY(particlesElectron, existNumElectronPerProcs);

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
        if (particlesSpecies[i].x < xminForProcs - buffer * device_dx) {
            particlesSpecies[i].isExist = false;
        }

        if (particlesSpecies[i].x > xmaxForProcs + buffer * device_dx) {
            particlesSpecies[i].isExist = false;
        }

        if (particlesSpecies[i].isMPISendLeftToRight) {
            unsigned long long particleIndex = atomicAdd(&(countForSendSpeciesLeftToRight[0]), 1);
            particlesSpecies[i].isMPISendLeftToRight = false;
            Particle sendParticle = particlesSpecies[i];
            if (sendParticle.x < device_xmin + buffer * device_dx) {
                sendParticle.x += device_xmax;
            }
            sendParticlesSpeciesLeftToRight[particleIndex] = sendParticle;
        }

        if (particlesSpecies[i].isMPISendRightToLeft) {
            unsigned long long particleIndex = atomicAdd(&(countForSendSpeciesRightToLeft[0]), 1);
            particlesSpecies[i].isMPISendRightToLeft = false;
            Particle sendParticle = particlesSpecies[i];
            if (sendParticle.x > device_xmax - buffer * device_dx) {
                sendParticle.x -= device_xmax;
            }
            sendParticlesSpeciesRightToLeft[particleIndex] = sendParticle;
        }
    }
}

void Boundary::periodicBoundaryParticleOfOneSpeciesX(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies
)
{
    device_countForSendSpeciesLeftToRight[0] = 0; 
    device_countForSendSpeciesRightToLeft[0] = 0; 
    countForSendSpeciesLeftToRight = 0;
    countForSendSpeciesRightToLeft = 0;
    countForRecvSpeciesLeftToRight = 0;
    countForRecvSpeciesRightToLeft = 0;

    float xminForProcs = xmin + (xmax - xmin) / mPIInfo.gridX * mPIInfo.localGridX;
    float xmaxForProcs = xmin + (xmax - xmin) / mPIInfo.gridX * (mPIInfo.localGridX + 1);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    periodicBoundaryParticleX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesLeftToRight.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesRightToLeft.data()), 
        thrust::raw_pointer_cast(device_countForSendSpeciesLeftToRight.data()), 
        thrust::raw_pointer_cast(device_countForSendSpeciesRightToLeft.data()), 
        existNumSpecies, 
        xminForProcs, xmaxForProcs, 
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

    countForSendSpeciesLeftToRight = device_countForSendSpeciesLeftToRight[0];
    countForSendSpeciesRightToLeft = device_countForSendSpeciesRightToLeft[0];

    sendrecv_particle_x(
        sendParticlesSpeciesLeftToRight, 
        sendParticlesSpeciesRightToLeft, 
        recvParticlesSpeciesLeftToRight, 
        recvParticlesSpeciesRightToLeft, 
        countForSendSpeciesLeftToRight, 
        countForSendSpeciesRightToLeft, 
        countForRecvSpeciesLeftToRight, 
        countForRecvSpeciesRightToLeft, 
        mPIInfo
    );

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
        if (particlesSpecies[i].y < yminForProcs - buffer * device_dy) {
            particlesSpecies[i].isExist = false;
        }

        if (particlesSpecies[i].y > ymaxForProcs + buffer * device_dy) {
            particlesSpecies[i].isExist = false;
        }

        if (particlesSpecies[i].isMPISendUpToDown) {
            unsigned long long particleIndex = atomicAdd(&(countForSendSpeciesUpToDown[0]), 1);
            particlesSpecies[i].isMPISendUpToDown = false;
            Particle sendParticle = particlesSpecies[i];
            if (sendParticle.y < device_ymin + buffer * device_dy) {
                sendParticle.y += device_ymax;
            }
            sendParticlesSpeciesUpToDown[particleIndex] = sendParticle;
        }

        if (particlesSpecies[i].isMPISendDownToUp) {
            unsigned long long particleIndex = atomicAdd(&(countForSendSpeciesDownToUp[0]), 1);
            particlesSpecies[i].isMPISendDownToUp = false;
            Particle sendParticle = particlesSpecies[i];
            if (sendParticle.y > device_ymax - buffer * device_dy) {
                sendParticle.y -= device_ymax;
            }
            sendParticlesSpeciesDownToUp[particleIndex] = sendParticle;
        }
    }
}

void Boundary::periodicBoundaryParticleOfOneSpeciesY(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies
)
{
    device_countForSendSpeciesUpToDown[0] = 0; 
    device_countForSendSpeciesDownToUp[0] = 0; 
    countForSendSpeciesUpToDown = 0;
    countForSendSpeciesDownToUp = 0;
    countForRecvSpeciesUpToDown = 0;
    countForRecvSpeciesDownToUp = 0;


    float yminForProcs = ymin + (ymax - ymin) / mPIInfo.gridY * mPIInfo.localGridY;
    float ymaxForProcs = ymin + (ymax - ymin) / mPIInfo.gridY * (mPIInfo.localGridY + 1);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    periodicBoundaryParticleY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesUpToDown.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesDownToUp.data()), 
        thrust::raw_pointer_cast(device_countForSendSpeciesUpToDown.data()), 
        thrust::raw_pointer_cast(device_countForSendSpeciesDownToUp.data()), 
        existNumSpecies, 
        yminForProcs, ymaxForProcs, 
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

    countForSendSpeciesUpToDown = device_countForSendSpeciesUpToDown[0];
    countForSendSpeciesDownToUp = device_countForSendSpeciesDownToUp[0];

    sendrecv_particle_y(
        sendParticlesSpeciesUpToDown, 
        sendParticlesSpeciesDownToUp, 
        recvParticlesSpeciesUpToDown, 
        recvParticlesSpeciesDownToUp, 
        countForSendSpeciesUpToDown, 
        countForSendSpeciesDownToUp, 
        countForRecvSpeciesUpToDown, 
        countForRecvSpeciesDownToUp, 
        mPIInfo
    );

    for (unsigned long long i = 0; i < countForRecvSpeciesUpToDown; i++) {
        particlesSpecies[existNumSpecies + i] = recvParticlesSpeciesUpToDown[i];
    }
    existNumSpecies += countForRecvSpeciesUpToDown;
    for (unsigned long long i = 0; i < countForRecvSpeciesDownToUp; i++) {
        particlesSpecies[existNumSpecies + i] = recvParticlesSpeciesDownToUp[i];
    }
    existNumSpecies += countForRecvSpeciesDownToUp;

}

//////////

void Boundary::periodicBoundaryBX(
    thrust::device_vector<MagneticField>& B
)
{
    
}


void Boundary::periodicBoundaryBY(
    thrust::device_vector<MagneticField>& B
)
{
    
}

//////////

void Boundary::periodicBoundaryEX(
    thrust::device_vector<ElectricField>& E
)
{
    
}


void Boundary::periodicBoundaryEY(
    thrust::device_vector<ElectricField>& E
)
{
    
}

//////////

void Boundary::periodicBoundaryCurrentX(
    thrust::device_vector<CurrentField>& current
)
{
    
}


void Boundary::periodicBoundaryCurrentY(
    thrust::device_vector<CurrentField>& current
)
{
    
}

