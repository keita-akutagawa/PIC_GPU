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


__global__ void conductingWallBoundaryParticleX_kernel(
    Particle* particlesSpecies, int totalNumSpecies
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < totalNumSpecies) {
        if (particlesSpecies[i].x < device_xmin) {
            particlesSpecies[i].x = 2.0f * device_xmin - particlesSpecies[i].x;
            particlesSpecies[i].vx = -1.0f * particlesSpecies[i].vx;
        }

        if (particlesSpecies[i].x > device_xmax) {
            particlesSpecies[i].x = 2.0f * device_xmax - particlesSpecies[i].x;
            particlesSpecies[i].vx = -1.0f * particlesSpecies[i].vx;
        }
    }
}

void Boundary::conductingWallBoundaryParticleX(
    thrust::device_vector<Particle>& particlesIon,
    thrust::device_vector<Particle>& particlesElectron
)
{
    dim3 threadsPerBlockForIon(256);
    dim3 blocksPerGridForIon((totalNumIon + threadsPerBlockForIon.x - 1) / threadsPerBlockForIon.x);

    conductingWallBoundaryParticleX_kernel<<<blocksPerGridForIon, threadsPerBlockForIon>>>(
        thrust::raw_pointer_cast(particlesIon.data()), totalNumIon
    );

    cudaDeviceSynchronize();

    dim3 threadsPerBlockForElectron(256);
    dim3 blocksPerGridForElectron((totalNumElectron + threadsPerBlockForElectron.x - 1) / threadsPerBlockForElectron.x);

    conductingWallBoundaryParticleX_kernel<<<blocksPerGridForElectron, threadsPerBlockForElectron>>>(
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

void Boundary::periodicBoundaryParticleY(
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


__global__ void conductingWallBoundaryParticleY_kernel(
    Particle* particlesSpecies, int totalNumSpecies
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < totalNumSpecies) {
        if (particlesSpecies[i].y < device_ymin) {
            particlesSpecies[i].y = 2.0f * device_ymin - particlesSpecies[i].y;
            particlesSpecies[i].vy = -1.0f * particlesSpecies[i].vy;
        }

        if (particlesSpecies[i].y > device_ymax) {
            particlesSpecies[i].y = 2.0f * device_ymax - particlesSpecies[i].y;
            particlesSpecies[i].vy = -1.0f * particlesSpecies[i].vy;
        }
    }
}

void Boundary::conductingWallBoundaryParticleY(
    thrust::device_vector<Particle>& particlesIon,
    thrust::device_vector<Particle>& particlesElectron
)
{
    dim3 threadsPerBlockForIon(256);
    dim3 blocksPerGridForIon((totalNumIon + threadsPerBlockForIon.x - 1) / threadsPerBlockForIon.x);

    conductingWallBoundaryParticleY_kernel<<<blocksPerGridForIon, threadsPerBlockForIon>>>(
        thrust::raw_pointer_cast(particlesIon.data()), totalNumIon
    );

    cudaDeviceSynchronize();

    dim3 threadsPerBlockForElectron(256);
    dim3 blocksPerGridForElectron((totalNumElectron + threadsPerBlockForElectron.x - 1) / threadsPerBlockForElectron.x);

    conductingWallBoundaryParticleY_kernel<<<blocksPerGridForElectron, threadsPerBlockForElectron>>>(
        thrust::raw_pointer_cast(particlesElectron.data()), totalNumElectron
    );

    cudaDeviceSynchronize();
}

//////////

void Boundary::periodicBoundaryBX(
    thrust::device_vector<MagneticField>& B
)
{
    return;
}

void Boundary::conductingWallBoundaryBX(
    thrust::device_vector<MagneticField>& B
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}

__global__ void symmetricBoundaryBX_kernel(
    MagneticField* B
)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < device_ny) {
        B[j + device_ny * 0].bX = B[j + device_ny * 1].bX;
        B[j + device_ny * 0].bY = 0.0f;
        B[j + device_ny * 0].bZ = 0.0f;

        B[j + device_ny * (device_nx - 1)].bX = B[j + device_ny * (device_nx - 2)].bX;
        B[j + device_ny * (device_nx - 2)].bY = 0.0f;
        B[j + device_ny * (device_nx - 1)].bY = 0.0f;
        B[j + device_ny * (device_nx - 2)].bZ = 0.0f;
        B[j + device_ny * (device_nx - 1)].bZ = 0.0f;
    }
}

void Boundary::symmetricBoundaryBX(
    thrust::device_vector<MagneticField>& B
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    symmetricBoundaryBX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data())
    );

    cudaDeviceSynchronize();
}


void Boundary::periodicBoundaryBY(
    thrust::device_vector<MagneticField>& B
)
{
    return;
}


__global__ void conductingWallBoundaryBY_kernel(
    MagneticField* B
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < device_nx) {
        B[0 + device_ny * i].bX = B[1 + device_ny * i].bX;
        B[1 + device_ny * i].bY = 0.0f;
        B[0 + device_ny * i].bY = 0.0f;
        B[0 + device_ny * i].bZ = B[1 + device_ny * i].bZ;

        B[device_ny - 1 + device_ny * i].bX = B[device_ny - 3 + device_ny * i].bX;
        B[device_ny - 1 + device_ny * i].bY = 0.0f;
        B[device_ny - 1 + device_ny * i].bZ = B[device_ny - 3 + device_ny * i].bZ;
    }
}


void Boundary::conductingWallBoundaryBY(
    thrust::device_vector<MagneticField>& B
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    conductingWallBoundaryBY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data())
    );

    cudaDeviceSynchronize();
}


void Boundary::symmetricBoundaryBY(
    thrust::device_vector<MagneticField>& B
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}

//////////


void Boundary::periodicBoundaryEX(
    thrust::device_vector<ElectricField>& E
)
{
    return;
}


void Boundary::conductingWallBoundaryEX(
    thrust::device_vector<ElectricField>& E
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}


__global__ void symmetricBoundaryEX_kernel(
    ElectricField* E
)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < device_ny) {
        E[j + device_ny * 0].eX = 0.0f;
        E[j + device_ny * 0].eY = E[j + device_ny * 1].eY;
        E[j + device_ny * 0].eZ = E[j + device_ny * 1].eZ;

        E[j + device_ny * (device_nx - 1)].eX = 0.0f;
        E[j + device_ny * (device_nx - 2)].eX = 0.0f;
        E[j + device_ny * (device_nx - 1)].eY = E[j + device_ny * (device_nx - 2)].eY;
        E[j + device_ny * (device_nx - 1)].eZ = E[j + device_ny * (device_nx - 2)].eZ;
    }
}


void Boundary::symmetricBoundaryEX(
    thrust::device_vector<ElectricField>& E
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    symmetricBoundaryEX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data())
    );

    cudaDeviceSynchronize();
}


void Boundary::periodicBoundaryEY(
    thrust::device_vector<ElectricField>& E
)
{
    return;
}


__global__ void conductingWallBoundaryEY_kernel(
    ElectricField* E
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < device_nx) {
        E[0 + device_ny * i].eX = E[2 + device_ny * i].eX;
        E[device_ny - 1 + device_ny * i].eX = E[device_ny - 2 + device_ny * i].eX;
        E[0 + device_ny * i].eY = -1.0f * E[1 + device_ny * i].eY;
        E[device_ny - 1 + device_ny * i].eY = 0.0f;
        E[device_ny - 2 + device_ny * i].eY = 0.0f;
        E[0 + device_ny * i].eZ = E[2 + device_ny * i].eZ;
        E[device_ny - 1 + device_ny * i].eZ = E[device_ny - 2 + device_ny * i].eZ;
    }
}


void Boundary::conductingWallBoundaryEY(
    thrust::device_vector<ElectricField>& E
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    conductingWallBoundaryEY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data())
    );

    cudaDeviceSynchronize();
}


void Boundary::symmetricBoundaryEY(
    thrust::device_vector<ElectricField>& E
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}

//////////

void Boundary::periodicBoundaryCurrentX(
    thrust::device_vector<CurrentField>& current
)
{
    return;
}


void Boundary::conductingWallBoundaryCurrentX(
    thrust::device_vector<CurrentField>& current
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}


__global__ void symmetricBoundaryCurrentX_kernel(
    CurrentField* current
)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < device_ny) {
        current[j + device_ny * 0].jX = 0.0f;
        current[j + device_ny * 0].jY = current[j + device_ny * 1].jY;
        current[j + device_ny * 0].jZ = current[j + device_ny * 1].jZ;
        current[j + device_ny * (device_nx - 1)].jX = 0.0f;
        current[j + device_ny * (device_nx - 1)].jY = current[j + device_ny * (device_nx - 2)].jY;
        current[j + device_ny * (device_nx - 1)].jZ = current[j + device_ny * (device_nx - 2)].jZ;
    }
}


void Boundary::symmetricBoundaryCurrentX(
    thrust::device_vector<CurrentField>& current
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    symmetricBoundaryCurrentX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data())
    );

    cudaDeviceSynchronize();
}


void Boundary::periodicBoundaryCurrentY(
    thrust::device_vector<CurrentField>& current
)
{
    return;
}


__global__ void conductingWallBoundaryCurrentY_kernel(
    CurrentField* current
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < device_nx) {
        current[0 + device_ny * i].jX = 0.0f;
        current[1 + device_ny * i].jX = 0.0f;
        current[0 + device_ny * i].jY = 0.0f;
        current[0 + device_ny * i].jZ = 0.0f;
        current[1 + device_ny * i].jZ = 0.0f;
        current[device_ny - 1 + device_ny * i].jX = -1.0f * current[device_ny - 2 + device_ny * i].jX;
        current[device_ny - 1 + device_ny * i].jY = 0.0f;
        current[device_ny - 2 + device_ny * i].jY = 0.0f;
        current[device_ny - 1 + device_ny * i].jZ = -1.0f * current[device_ny - 2 + device_ny * i].jZ;
    }
}


void Boundary::conductingWallBoundaryCurrentY(
    thrust::device_vector<CurrentField>& current
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    conductingWallBoundaryCurrentY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data())
    );

    cudaDeviceSynchronize();
}


void Boundary::symmetricBoundaryCurrentY(
    thrust::device_vector<CurrentField>& current
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}

