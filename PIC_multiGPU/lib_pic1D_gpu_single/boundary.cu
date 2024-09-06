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
    int* countForSendSpeciesLeftToRight, 
    int* countForSendSpeciesRightToLeft, 
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
        Particle sendParticle;

        if (xminForProcs <= particlesSpecies[i].x && particlesSpecies[i].x < xminForProcs + device_dx) {
            index = atomicAdd(&(countForSendSpeciesLeftToRight[0]), 1);
            sendParticle = particlesSpecies[i];
            if (sendParticle.x < device_xmin + device_dx) {
                sendParticle.x += device_xmax;
            }
            sendParticlesSpeciesLeftToRight[index] = sendParticle;
        }

        if (xmaxForProcs - device_dx < particlesSpecies[i].x && particlesSpecies[i].x <= xmaxForProcs) {
            index = atomicAdd(&(countForSendSpeciesRightToLeft[0]), 1);
            sendParticle = particlesSpecies[i];
            if (sendParticle.x > device_xmax - device_dx) {
                sendParticle.x -= device_xmax;
            }
            sendParticlesSpeciesRightToLeft[index] = sendParticle;
        }
    }
}


void Boundary::periodicBoundaryParticleX(
    thrust::device_vector<Particle>& particlesIon,
    thrust::device_vector<Particle>& particlesElectron, 
    MPIInfo& mPIInfo
)
{
    thrust::device_vector<int> device_countForSendIonLeftToRight(1);
    thrust::device_vector<int> device_countForSendIonRightToLeft(1);
    thrust::device_vector<int> device_countForSendElectronLeftToRight(1);
    thrust::device_vector<int> device_countForSendElectronRightToLeft(1);
    device_countForSendIonLeftToRight[0] = 0; device_countForSendIonRightToLeft[0] = 0;
    device_countForSendElectronLeftToRight[0] = 0; device_countForSendElectronRightToLeft[0] = 0;
    int countForSendIonLeftToRight;
    int countForSendIonRightToLeft;
    int countForSendElectronLeftToRight;
    int countForSendElectronRightToLeft;


    dim3 threadsPerBlockForIon(256);
    dim3 blocksPerGridForIon((mPIInfo.existNumIonPerProcs + threadsPerBlockForIon.x - 1) / threadsPerBlockForIon.x);

    periodicBoundaryParticleX_kernel<<<blocksPerGridForIon, threadsPerBlockForIon>>>(
        thrust::raw_pointer_cast(particlesIon.data()), 
        thrust::raw_pointer_cast(sendParticlesIonLeftToRight.data()), 
        thrust::raw_pointer_cast(sendParticlesIonRightToLeft.data()), 
        thrust::raw_pointer_cast(device_countForSendIonLeftToRight.data()), 
        thrust::raw_pointer_cast(device_countForSendIonRightToLeft.data()), 
        mPIInfo.existNumIonPerProcs, 
        xmin + (xmax - xmin) / mPIInfo.procs * mPIInfo.rank, 
        xmin + (xmax - xmin) / mPIInfo.procs * (mPIInfo.rank + 1)
    );

    cudaDeviceSynchronize();

    dim3 threadsPerBlockForElectron(256);
    dim3 blocksPerGridForElectron((mPIInfo.existNumElectronPerProcs + threadsPerBlockForElectron.x - 1) / threadsPerBlockForElectron.x);

    periodicBoundaryParticleX_kernel<<<blocksPerGridForElectron, threadsPerBlockForElectron>>>(
        thrust::raw_pointer_cast(particlesElectron.data()), 
        thrust::raw_pointer_cast(sendParticlesElectronLeftToRight.data()), 
        thrust::raw_pointer_cast(sendParticlesElectronRightToLeft.data()), 
        thrust::raw_pointer_cast(device_countForSendElectronLeftToRight.data()), 
        thrust::raw_pointer_cast(device_countForSendElectronRightToLeft.data()), 
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
    countForSendIonLeftToRight = device_countForSendIonLeftToRight[0];
    countForSendIonRightToLeft = device_countForSendIonRightToLeft[0];
    countForSendElectronLeftToRight = device_countForSendElectronLeftToRight[0];
    countForSendElectronRightToLeft = device_countForSendElectronRightToLeft[0];


    sendrecv_particle(
        host_sendParticlesIonLeftToRight, 
        host_sendParticlesIonRightToLeft, 
        host_recvParticlesIonLeftToRight, 
        host_recvParticlesIonRightToLeft, 
        countForSendIonLeftToRight, 
        countForSendIonRightToLeft, 
        mPIInfo
    );

    sendrecv_particle(
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


