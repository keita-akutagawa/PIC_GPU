#include "boundary.hpp"


Boundary::Boundary(MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo), 

      sendParticlesSpeciesLeftToRight     (numberDensityIon * 10 * mPIInfo.localSizeY), 
      sendParticlesSpeciesRightToLeft     (numberDensityIon * 10 * mPIInfo.localSizeY), 
      host_sendParticlesSpeciesLeftToRight(numberDensityIon * 10 * mPIInfo.localSizeY), 
      host_sendParticlesSpeciesRightToLeft(numberDensityIon * 10 * mPIInfo.localSizeY), 
      host_recvParticlesSpeciesLeftToRight(numberDensityIon * 10 * mPIInfo.localSizeY), 
      host_recvParticlesSpeciesRightToLeft(numberDensityIon * 10 * mPIInfo.localSizeY), 

      sendParticlesSpeciesUpToDown     (numberDensityIon * mPIInfo.localSizeX * 10), 
      sendParticlesSpeciesDownToUp     (numberDensityIon * mPIInfo.localSizeX * 10), 
      host_sendParticlesSpeciesUpToDown(numberDensityIon * mPIInfo.localSizeX * 10), 
      host_sendParticlesSpeciesDownToUp(numberDensityIon * mPIInfo.localSizeX * 10), 
      host_recvParticlesSpeciesUpToDown(numberDensityIon * mPIInfo.localSizeX * 10), 
      host_recvParticlesSpeciesDownToUp(numberDensityIon * mPIInfo.localSizeX * 10), 

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
    int operator()(const Particle& p) const {
        return p.isExist ? 1 : 0;
    }
};


void Boundary::periodicBoundaryParticleXY(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron
)
{
    periodicBoundaryParticleOfOneSpeciesX(particlesIon, mPIInfo.existNumIonPerProcs);
    periodicBoundaryParticleOfOneSpeciesX(particlesElectron, mPIInfo.existNumElectronPerProcs);

    MPI_Barrier(MPI_COMM_WORLD);

    periodicBoundaryParticleOfOneSpeciesY(particlesIon, mPIInfo.existNumIonPerProcs);
    periodicBoundaryParticleOfOneSpeciesY(particlesElectron, mPIInfo.existNumElectronPerProcs);

    MPI_Barrier(MPI_COMM_WORLD);
}


__global__ void periodicBoundaryParticleX_kernel(
    Particle* particlesSpecies, 
    Particle* sendParticlesSpeciesLeftToRight, 
    Particle* sendParticlesSpeciesRightToLeft, 
    unsigned long long* countForSendSpeciesLeftToRight, 
    unsigned long long* countForSendSpeciesRightToLeft, 
    const unsigned long long existNumSpecies, 
    const float xminForProcs, const float xmaxForProcs
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (particlesSpecies[i].x <= xminForProcs) {
            particlesSpecies[i].isExist = false;
        }

        if (particlesSpecies[i].x >= xmaxForProcs) {
            particlesSpecies[i].isExist = false;
        }


        int index;
        Particle sendParticle;

        if (xminForProcs < particlesSpecies[i].x && particlesSpecies[i].x <= xminForProcs + device_dx) {
            index = atomicAdd(&(countForSendSpeciesLeftToRight[0]), 1);
            sendParticle = particlesSpecies[i];
            if (sendParticle.x <= device_xmin + device_dx) {
                sendParticle.x += device_xmax;
            }
            sendParticlesSpeciesLeftToRight[index] = sendParticle;
        }

        if (xmaxForProcs - device_dx <= particlesSpecies[i].x && particlesSpecies[i].x < xmaxForProcs) {
            index = atomicAdd(&(countForSendSpeciesRightToLeft[0]), 1);
            sendParticle = particlesSpecies[i];
            if (sendParticle.x >= device_xmax - device_dx) {
                sendParticle.x -= device_xmax;
            }
            sendParticlesSpeciesRightToLeft[index] = sendParticle;
        }
    }
}

void Boundary::periodicBoundaryParticleOfOneSpeciesX(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies
)
{
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
        xminForProcs, xmaxForProcs 
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

    for (int i = 0; i < countForRecvSpeciesLeftToRight; i++) {
        particlesSpecies[existNumSpecies + i] = host_recvParticlesSpeciesLeftToRight[i];
    }
    existNumSpecies += countForRecvSpeciesLeftToRight;
    for (int i = 0; i < countForRecvSpeciesRightToLeft; i++) {
        particlesSpecies[existNumSpecies + i] = host_recvParticlesSpeciesRightToLeft[i];
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
    const float yminForProcs, const float ymaxForProcs
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (particlesSpecies[i].y <= yminForProcs) {
            particlesSpecies[i].isExist = false;
        }

        if (particlesSpecies[i].y >= ymaxForProcs) {
            particlesSpecies[i].isExist = false;
        }


        int index;
        Particle sendParticle;

        if (yminForProcs < particlesSpecies[i].y && particlesSpecies[i].y <= yminForProcs + device_dy) {
            index = atomicAdd(&(countForSendSpeciesUpToDown[0]), 1);
            sendParticle = particlesSpecies[i];
            if (sendParticle.y <= device_ymin + device_dy) {
                sendParticle.y += device_ymax;
            }
            sendParticlesSpeciesUpToDown[index] = sendParticle;
        }

        if (ymaxForProcs - device_dy <= particlesSpecies[i].y && particlesSpecies[i].y < ymaxForProcs) {
            index = atomicAdd(&(countForSendSpeciesDownToUp[0]), 1);
            sendParticle = particlesSpecies[i];
            if (sendParticle.y >= device_ymax - device_dy) {
                sendParticle.y -= device_ymax;
            }
            sendParticlesSpeciesDownToUp[index] = sendParticle;
        }
    }
}

void Boundary::periodicBoundaryParticleOfOneSpeciesY(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies
)
{
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
        yminForProcs, ymaxForProcs 
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

    for (int i = 0; i < countForRecvSpeciesUpToDown; i++) {
        particlesSpecies[existNumSpecies + i] = host_recvParticlesSpeciesUpToDown[i];
    }
    existNumSpecies += countForRecvSpeciesUpToDown;
    for (int i = 0; i < countForRecvSpeciesDownToUp; i++) {
        particlesSpecies[existNumSpecies + i] = host_recvParticlesSpeciesDownToUp[i];
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

