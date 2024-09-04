#include "boundary.hpp"


Boundary::Boundary()
    : sendrecvParticlesIonLeft(numberDensityIon * 10), 
      sendrecvParticlesElectronLeft(numberDensityIon * 10), 
      sendrecvParticlesIonRight(numberDensityIon * 10), 
      sendrecvParticlesElectronRight(numberDensityIon * 10)
{
}


__global__ void periodicBoundaryParticleX_kernel(
    Particle* particlesSpecies, 
    Particle* sendrecvParticlesSpeciesLeftToRight, 
    Particle* sendrecvParticlesSpeciesRightToLeft, 
    int& countForSendrecvSpeciesLeftToRight, 
    int& countForSendrecvSpeciesRightToLeft, 
    const int existNumSpecies, 
    const float xminForProcs, 
    const float xmaxForProcs
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
            index = atomicAdd(&countForSendrecvSpeciesLeftToRight, 1);
            sendrecvParticlesSpeciesLeftToRight[index] = particlesSpecies[i];
        }

        if (xmaxForProcs - device_dx < particlesSpecies[i].x && particlesSpecies[i].x <= xmaxForProcs) {
            index = atomicAdd(&countForSendrecvSpeciesRightToLeft, 1);
            sendrecvParticlesSpeciesRightToLeft[index] = particlesSpecies[i];
        }
    }
}


void Boundary::periodicBoundaryParticleX(
    thrust::device_vector<Particle>& particlesIon,
    thrust::device_vector<Particle>& particlesElectron, 
    MPIInfo& mPIInfo
)
{
    int countForSendrecvIonLeftToRight, countForSendrecvIonRightToLeft;
    int countForSendrecvElectronLeftToRight, countForSendrecvElectronRightToLeft;
    countForSendrecvIonLeftToRight = 0; countForSendrecvIonRightToLeft = 0;
    countForSendrecvElectronLeftToRight = 0; countForSendrecvElectronRightToLeft = 0; 

    dim3 threadsPerBlockForIon(256);
    dim3 blocksPerGridForIon((mPIInfo.existNumIonPerProcs + threadsPerBlockForIon.x - 1) / threadsPerBlockForIon.x);

    periodicBoundaryParticleX_kernel<<<blocksPerGridForIon, threadsPerBlockForIon>>>(
        thrust::raw_pointer_cast(particlesIon.data()), 
        thrust::raw_pointer_cast(sendrecvParticlesIonLeftToRight.data()), 
        thrust::raw_pointer_cast(sendrecvParticlesIonRightToLeft.data()), 
        mPIInfo.existNumIonPerProcs, 
        xmin + (xmax - xmin) / mPIInfo.procs * mPIInfo.rank, 
        xmin + (xmax - xmin) / mPIInfo.procs * (mPIInfo.rank + 1)
    );

    cudaDeviceSynchronize();

    dim3 threadsPerBlockForElectron(256);
    dim3 blocksPerGridForElectron((mPIInfo.existNumElectronPerProcs + threadsPerBlockForElectron.x - 1) / threadsPerBlockForElectron.x);

    periodicBoundaryParticleX_kernel<<<blocksPerGridForElectron, threadsPerBlockForElectron>>>(
        thrust::raw_pointer_cast(particlesElectron.data()), 
        thrust::raw_pointer_cast(sendrecvParticlesIonLeftToRight.data()), 
        thrust::raw_pointer_cast(sendrecvParticlesIonRightToLeft.data()),  
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


