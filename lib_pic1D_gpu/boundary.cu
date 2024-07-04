#include "boundary.hpp"


__global__ void periodicBoundaryParticleIonX_kernel(
    Particle* particlesIon
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < device_totalNumIon) {
        if (particlesIon[i].x < device_xmin) {
            particlesIon[i].x += device_xmax - device_xmin;
        }

        if (particlesIon[i].x > device_xmax) {
            particlesIon[i].x -= device_xmax - device_xmin;
        }
    }
}

__global__ void periodicBoundaryParticleElectronX_kernel(
    Particle* particlesElectron
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < device_totalNumElectron) {
        if (particlesElectron[i].x < device_xmin) {
            particlesElectron[i].x += device_xmax - device_xmin;
        }

        if (particlesElectron[i].x > device_xmax) {
            particlesElectron[i].x -= device_xmax - device_xmin;
        }
    }
}


void Boundary::periodicBoundaryParticleX(
    thrust::device_vector<Particle>& particlesIon,
    thrust::device_vector<Particle>& particlesElectron
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((totalNumParticles + threadsPerBlock.x - 1) / threadsPerBlock.x);

    periodicBoundaryParticleIonX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesIon.data())
    );

    cudaDeviceSynchronize();

    periodicBoundaryParticleElectronX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesElectron.data())
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


