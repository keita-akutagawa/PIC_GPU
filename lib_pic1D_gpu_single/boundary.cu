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
    MagneticField* magneticField
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 1) {
        magneticField[0] = magneticField[device_nx - 2];
        magneticField[device_nx - 1] = magneticField[1];
    }
}

void Boundary::periodicBoundaryBX(
    thrust::device_vector<MagneticField>& magneticField
)
{
    dim3 threadsPerBlock(1);
    dim3 blocksPerGrid(1);

    periodicBoundaryBX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(magneticField.data())
    );

    cudaDeviceSynchronize();
}


__global__ void periodicBoundaryEX_kernel(
    ElectricField* electricField
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 1) {
        electricField[0] = electricField[device_nx - 2];
        electricField[device_nx - 1] = electricField[1];
    }
}

void Boundary::periodicBoundaryEX(
    thrust::device_vector<ElectricField>& electricField
)
{
    dim3 threadsPerBlock(1);
    dim3 blocksPerGrid(1);

    periodicBoundaryEX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(electricField.data())
    );

    cudaDeviceSynchronize();
}


__global__ void periodicBoundaryCurrentX_kernel(
    CurrentField* currentField
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 1) {
        currentField[0] = currentField[device_nx - 2];
        currentField[device_nx - 1] = currentField[1];
    }
}

void Boundary::periodicBoundaryCurrentX(
    thrust::device_vector<CurrentField>& currentField
)
{
    dim3 threadsPerBlock(1);
    dim3 blocksPerGrid(1);

    periodicBoundaryCurrentX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(currentField.data())
    );

    cudaDeviceSynchronize();
}


