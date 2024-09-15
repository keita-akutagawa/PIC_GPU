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


void Boundary::periodicBoundaryParticleX(
    thrust::device_vector<Particle>& particlesIon,
    thrust::device_vector<Particle>& particlesElectron, 
    MPIInfo& mPIInfo
)
{
    double xminForProcs = xmin + (xmax - xmin) / mPIInfo.procs * mPIInfo.rank;
    double xmaxForProcs = xmin + (xmax - xmin) / mPIInfo.procs * (mPIInfo.rank + 1);

    thrust::device_vector<int> device_countForSendIonLeftToRight(1, 0);
    thrust::device_vector<int> device_countForSendIonRightToLeft(1, 0);
    thrust::device_vector<int> device_countForSendElectronLeftToRight(1, 0);
    thrust::device_vector<int> device_countForSendElectronRightToLeft(1, 0);
    int countForSendIonLeftToRight = 0;
    int countForSendIonRightToLeft = 0;
    int countForSendElectronLeftToRight = 0;
    int countForSendElectronRightToLeft = 0;
    int countForRecvIonLeftToRight = 0;
    int countForRecvIonRightToLeft = 0;
    int countForRecvElectronLeftToRight = 0;
    int countForRecvElectronRightToLeft = 0;


    dim3 threadsPerBlockForIon(256);
    dim3 blocksPerGridForIon((mPIInfo.existNumIonPerProcs + threadsPerBlockForIon.x - 1) / threadsPerBlockForIon.x);

    periodicBoundaryParticleX_kernel<<<blocksPerGridForIon, threadsPerBlockForIon>>>(
        thrust::raw_pointer_cast(particlesIon.data()), 
        thrust::raw_pointer_cast(sendParticlesIonLeftToRight.data()), 
        thrust::raw_pointer_cast(sendParticlesIonRightToLeft.data()), 
        thrust::raw_pointer_cast(device_countForSendIonLeftToRight.data()), 
        thrust::raw_pointer_cast(device_countForSendIonRightToLeft.data()), 
        mPIInfo.existNumIonPerProcs, 
        xminForProcs, xmaxForProcs 
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
        xminForProcs, xmaxForProcs
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


    MPI_Barrier(MPI_COMM_WORLD);

    sendrecv_particle(
        host_sendParticlesIonLeftToRight, 
        host_sendParticlesIonRightToLeft, 
        host_recvParticlesIonLeftToRight, 
        host_recvParticlesIonRightToLeft, 
        countForSendIonLeftToRight, 
        countForSendIonRightToLeft, 
        countForRecvIonLeftToRight, 
        countForRecvIonRightToLeft, 
        mPIInfo
    );

    sendrecv_particle(
        host_sendParticlesElectronLeftToRight, 
        host_sendParticlesElectronRightToLeft, 
        host_recvParticlesElectronLeftToRight, 
        host_recvParticlesElectronRightToLeft, 
        countForSendElectronLeftToRight, 
        countForSendElectronRightToLeft, 
        countForRecvElectronLeftToRight, 
        countForRecvElectronRightToLeft, 
        mPIInfo
    );

    MPI_Barrier(MPI_COMM_WORLD);


    for (int i = 0; i < countForRecvIonLeftToRight; i++) {
        particlesIon[mPIInfo.existNumIonPerProcs + i] = host_recvParticlesIonLeftToRight[i];
    }
    mPIInfo.existNumIonPerProcs += countForRecvIonLeftToRight;
    for (int i = 0; i < countForRecvIonRightToLeft; i++) {
        particlesIon[mPIInfo.existNumIonPerProcs + i] = host_recvParticlesIonRightToLeft[i];
    }
    mPIInfo.existNumIonPerProcs += countForRecvIonRightToLeft;

    for (int i = 0; i < countForRecvElectronLeftToRight; i++) {
        particlesElectron[mPIInfo.existNumElectronPerProcs + i] = host_recvParticlesElectronLeftToRight[i];
    }
    mPIInfo.existNumElectronPerProcs += countForRecvElectronLeftToRight;
    for (int i = 0; i < countForRecvElectronRightToLeft; i++) {
        particlesElectron[mPIInfo.existNumElectronPerProcs + i] = host_recvParticlesElectronRightToLeft[i];
    }
    mPIInfo.existNumElectronPerProcs += countForRecvElectronRightToLeft;


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


