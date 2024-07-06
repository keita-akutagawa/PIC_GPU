#include "boundary.hpp"


__global__ void periodicBoundaryParticleX_kernel(
    Particle* particlesSpecies, int totalNumSpecies
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < totalNumSpecies) {
        if (particlesSpecies[i].x < device_xmin) {
            particlesSpecies[i].x += device_xmax - device_xmin;
        }

        if (particlesSpecies[i].x > device_xmax) {
            particlesSpecies[i].x -= device_xmax - device_xmin;
        }
    }
}


void Boundary::periodicBoundaryParticleX(
    thrust::device_vector<Particle>& particlesIon,
    thrust::device_vector<Particle>& particlesElectron
)
{
    dim3 threadsPerBlockForIon(256);
    dim3 blocksPerGridForIon((totalNumIon + threadsPerBlockForIon.x - 1) / threadsPerBlockForIon.x);

    periodicBoundaryParticleX_kernel<<<blocksPerGridForIon, threadsPerBlockForIon>>>(
        thrust::raw_pointer_cast(particlesIon.data()), totalNumIon
    );

    cudaDeviceSynchronize();

    dim3 threadsPerBlockForElectron(256);
    dim3 blocksPerGridForElectron((totalNumElectron + threadsPerBlockForElectron.x - 1) / threadsPerBlockForElectron.x);

    periodicBoundaryParticleX_kernel<<<blocksPerGridForElectron, threadsPerBlockForElectron>>>(
        thrust::raw_pointer_cast(particlesElectron.data()), totalNumElectron
    );

    cudaDeviceSynchronize();
}

//////////

__global__ void periodicBoundaryBX_kernel(
    MagneticField* B
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 1) {
        B[0] = B[device_nx - 2];
        B[device_nx - 1] = B[1];
    }
}

void Boundary::periodicBoundaryBX(
    thrust::device_vector<MagneticField>& B
)
{
    dim3 threadsPerBlock(1);
    dim3 blocksPerGrid(1);

    periodicBoundaryBX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data())
    );

    cudaDeviceSynchronize();
}


__global__ void periodicBoundaryEX_kernel(
    ElectricField* E
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 1) {
        E[0] = E[device_nx - 2];
        E[device_nx - 1] = E[1];
    }
}

void Boundary::periodicBoundaryEX(
    thrust::device_vector<ElectricField>& E
)
{
    dim3 threadsPerBlock(1);
    dim3 blocksPerGrid(1);

    periodicBoundaryEX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data())
    );

    cudaDeviceSynchronize();
}


__global__ void periodicBoundaryCurrentX_kernel(
    CurrentField* current
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 1) {
        current[0] = current[device_nx - 2];
        current[device_nx - 1] = current[1];
    }
}

void Boundary::periodicBoundaryCurrentX(
    thrust::device_vector<CurrentField>& current
)
{
    dim3 threadsPerBlock(1);
    dim3 blocksPerGrid(1);

    periodicBoundaryCurrentX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data())
    );

    cudaDeviceSynchronize();
}


