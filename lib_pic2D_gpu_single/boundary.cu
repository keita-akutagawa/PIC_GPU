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


__global__ void periodicBoundaryParticleY_kernel(
    Particle* particlesSpecies, int totalNumSpecies
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < totalNumSpecies) {
        if (particlesSpecies[i].y < device_ymin) {
            particlesSpecies[i].y += device_ymax - device_ymin;
        }

        if (particlesSpecies[i].y > device_ymax) {
            particlesSpecies[i].y -= device_ymax - device_ymin;
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

    periodicBoundaryParticleY_kernel<<<blocksPerGridForIon, threadsPerBlockForIon>>>(
        thrust::raw_pointer_cast(particlesIon.data()), totalNumIon
    );

    cudaDeviceSynchronize();

    dim3 threadsPerBlockForElectron(256);
    dim3 blocksPerGridForElectron((totalNumElectron + threadsPerBlockForElectron.x - 1) / threadsPerBlockForElectron.x);

    periodicBoundaryParticleY_kernel<<<blocksPerGridForElectron, threadsPerBlockForElectron>>>(
        thrust::raw_pointer_cast(particlesElectron.data()), totalNumElectron
    );

    cudaDeviceSynchronize();
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


