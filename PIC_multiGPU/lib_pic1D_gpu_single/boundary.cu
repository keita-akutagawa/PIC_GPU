#include "boundary.hpp"


Boundary::Boundary()
    : sendParticlesIonLeftToRight(numberDensityIon * 10), 
      sendParticlesIonRightToLeft(numberDensityIon * 10), 
      sendParticlesElectronLeftToRight(numberDensityElectron * 10), 
      sendParticlesElectronRightToLeft(numberDensityElectron * 10), 

      host_sendParticlesIonLeftToRight(numberDensityIon * 10), 
      host_sendParticlesIonRightToLeft(numberDensityIon * 10), 
      host_sendParticlesElectronLeftToRight(numberDensityElectron * 10), 
      host_sendParticlesElectronRightToLeft(numberDensityElectron * 10), 

      host_recvParticlesIonLeftToRight(numberDensityIon * 10), 
      host_recvParticlesIonRightToLeft(numberDensityIon * 10), 
      host_recvParticlesElectronLeftToRight(numberDensityElectron * 10), 
      host_recvParticlesElectronRightToLeft(numberDensityElectron * 10)
{
}


__global__ void periodicBoundaryParticleX_kernel(
    Particle* particlesSpecies, 
    Particle* sendParticlesSpeciesLeftToRight, 
    Particle* sendParticlesSpeciesRightToLeft, 
    int& countForSendSpeciesLeftToRight, 
    int& countForSendSpeciesRightToLeft, 
    const int existNumSpecies, 
    const double xminForProcs, 
    const double xmaxForProcs
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (particlesSpecies[i].x < xminForProcs) {
            particlesSpecies[i].isExist = false;
        }

        if (particlesSpecies[i].x > xmaxForProcs) {
            particlesSpecies[i].isExist = false;
        }


        int index;

        if (xminForProcs <= particlesSpecies[i].x && particlesSpecies[i].x < xminForProcs + device_dx) {
            index = atomicAdd(&countForSendSpeciesLeftToRight, 1);
            sendParticlesSpeciesLeftToRight[index] = particlesSpecies[i];
        }

        if (xmaxForProcs - device_dx < particlesSpecies[i].x && particlesSpecies[i].x <= xmaxForProcs) {
            index = atomicAdd(&countForSendSpeciesRightToLeft, 1);
            sendParticlesSpeciesRightToLeft[index] = particlesSpecies[i];
        }
    }
}


void Boundary::periodicBoundaryParticleX(
    thrust::device_vector<Particle>& particlesIon,
    thrust::device_vector<Particle>& particlesElectron, 
    MPIInfo& mPIInfo
)
{
    int countForSendIonLeftToRight, countForSendIonRightToLeft;
    int countForSendElectronLeftToRight, countForSendElectronRightToLeft;
    countForSendIonLeftToRight = 0; countForSendIonRightToLeft = 0;
    countForSendElectronLeftToRight = 0; countForSendElectronRightToLeft = 0; 

    dim3 threadsPerBlockForIon(256);
    dim3 blocksPerGridForIon((mPIInfo.existNumIonPerProcs + threadsPerBlockForIon.x - 1) / threadsPerBlockForIon.x);

    periodicBoundaryParticleX_kernel<<<blocksPerGridForIon, threadsPerBlockForIon>>>(
        thrust::raw_pointer_cast(particlesIon.data()), 
        thrust::raw_pointer_cast(sendParticlesIonLeftToRight.data()), 
        thrust::raw_pointer_cast(sendParticlesIonRightToLeft.data()), 
        countForSendIonLeftToRight, 
        countForSendIonRightToLeft, 
        mPIInfo.existNumIonPerProcs, 
        xmin + (xmax - xmin) / mPIInfo.procs * mPIInfo.rank, 
        xmin + (xmax - xmin) / mPIInfo.procs * (mPIInfo.rank + 1)
    );

    cudaDeviceSynchronize();

    dim3 threadsPerBlockForElectron(256);
    dim3 blocksPerGridForElectron((mPIInfo.existNumElectronPerProcs + threadsPerBlockForElectron.x - 1) / threadsPerBlockForElectron.x);

    periodicBoundaryParticleX_kernel<<<blocksPerGridForElectron, threadsPerBlockForElectron>>>(
        thrust::raw_pointer_cast(particlesElectron.data()), 
        thrust::raw_pointer_cast(sendParticlesIonLeftToRight.data()), 
        thrust::raw_pointer_cast(sendParticlesIonRightToLeft.data()),  
        countForSendElectronLeftToRight, 
        countForSendElectronRightToLeft, 
        mPIInfo.existNumElectronPerProcs, 
        xmin + (xmax - xmin) / mPIInfo.procs * mPIInfo.rank, 
        xmin + (xmax - xmin) / mPIInfo.procs * (mPIInfo.rank + 1)
    );

    cudaDeviceSynchronize();


    mPIInfo.existNumIonPerProcs = thrust::transform_reduce(
        particlesIon.begin(),
        particlesIon.end(),
        IsExistTransform(), 
        0,               
        thrust::plus<int>()
    );

    cudaDeviceSynchronize();

    mPIInfo.existNumElectronPerProcs = thrust::transform_reduce(
        particlesElectron.begin(),
        particlesElectron.end(),
        IsExistTransform(), 
        0,               
        thrust::plus<int>()
    );

    cudaDeviceSynchronize();


    thrust::partition(
        particlesIon.begin(), particlesIon.end(), 
        [] __device__ (const Particle& p) { return p.isExist; }
    );
    cudaDeviceSynchronize();
    
    thrust::partition(
        particlesElectron.begin(), particlesElectron.end(), 
        [] __device__ (const Particle& p) { return p.isExist; }
    );
    cudaDeviceSynchronize();


    host_sendParticlesIonLeftToRight = sendParticlesIonLeftToRight;
    host_sendParticlesIonRightToLeft = sendParticlesIonRightToLeft;
    host_sendParticlesElectronLeftToRight = sendParticlesElectronLeftToRight;
    host_sendParticlesElectronRightToLeft = sendParticlesElectronRightToLeft;

    sendrecv_particle(
        particlesIon, 
        host_sendParticlesIonLeftToRight, 
        host_sendParticlesIonRightToLeft, 
        host_recvParticlesIonLeftToRight, 
        host_recvParticlesIonRightToLeft, 
        countForSendIonLeftToRight, 
        countForSendIonRightToLeft, 
        mPIInfo
    );

    sendrecv_particle(
        particlesElectron, 
        host_sendParticlesElectronLeftToRight, 
        host_sendParticlesElectronRightToLeft, 
        host_recvParticlesElectronLeftToRight, 
        host_recvParticlesElectronRightToLeft, 
        countForSendElectronLeftToRight, 
        countForSendElectronRightToLeft, 
        mPIInfo
    );


    for (int i = 0; i < countForSendIonLeftToRight; i++) {
        particlesIon[mPIInfo.existNumIonPerProcs + i] = host_recvParticlesIonLeftToRight[i];
    }
    mPIInfo.existNumIonPerProcs += countForSendIonLeftToRight;
    for (int i = 0; i < countForSendIonRightToLeft; i++) {
        particlesIon[mPIInfo.existNumIonPerProcs + i] = host_recvParticlesIonRightToLeft[i];
    }
    mPIInfo.existNumIonPerProcs += countForSendIonRightToLeft;

    for (int i = 0; i < countForSendElectronLeftToRight; i++) {
        particlesElectron[mPIInfo.existNumElectronPerProcs + i] = host_recvParticlesElectronLeftToRight[i];
    }
    mPIInfo.existNumElectronPerProcs += countForSendElectronLeftToRight;
    for (int i = 0; i < countForSendElectronRightToLeft; i++) {
        particlesElectron[mPIInfo.existNumElectronPerProcs + i] = host_recvParticlesElectronRightToLeft[i];
    }
    mPIInfo.existNumElectronPerProcs += countForSendElectronRightToLeft;


    if (mPIInfo.existNumIonPerProcs > mPIInfo.totalNumIonPerProcs) printf("ERROR!\n");
    if (mPIInfo.existNumElectronPerProcs > mPIInfo.totalNumElectronPerProcs) printf("ERROR!\n");
}

//////////

void Boundary::periodicBoundaryBX(
    thrust::device_vector<MagneticField>& B
)
{
}


void Boundary::periodicBoundaryEX(
    thrust::device_vector<ElectricField>& E
)
{
}


void Boundary::periodicBoundaryCurrentX(
    thrust::device_vector<CurrentField>& current
)
{
}


