#include "field_solver.hpp"


FieldSolver::FieldSolver(MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo)
{
}


__global__ void timeEvolutionB_kernel(
    MagneticField* B, const ElectricField* E, const float dt, 
    int localSizeX, int localSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeX - 1 && j < localSizeY - 1) {
        int index = j + i * localSizeY;

        B[index].bX += -(E[index + 1].eZ - E[index].eZ) / device_dy * dt;
        B[index].bY += (E[index + localSizeY].eZ - E[index].eZ) / device_dx * dt;
        B[index].bZ += (-(E[index + localSizeY].eY - E[index].eY) / device_dx
                     + (E[index + 1].eX - E[index].eX) / device_dy) * dt;
    }
}

void FieldSolver::timeEvolutionB(
    thrust::device_vector<MagneticField>& B, 
    const thrust::device_vector<ElectricField>& E, 
    const float dt
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (mPIInfo.localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    timeEvolutionB_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data()), 
        dt, 
        mPIInfo.localSizeX, mPIInfo.localSizeY
    );
    cudaDeviceSynchronize();
}



__global__ void timeEvolutionE_kernel(
    ElectricField* E, const MagneticField* B, const CurrentField* current, 
    const float dt, 
    int localSizeX, int localSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < localSizeX) && (0 < j) && (j < localSizeY)) {
        int index = j + i * localSizeY;

        E[index].eX += (-current[index].jX / device_epsilon0
                     + device_c * device_c * (B[index].bZ - B[index - 1].bZ) / device_dy) * dt;
        E[index].eY += (-current[index].jY / device_epsilon0 
                     - device_c * device_c * (B[index].bZ - B[index - localSizeY].bZ) / device_dx) * dt;
        E[index].eZ += (-current[index].jZ / device_epsilon0 
                     + device_c * device_c * ((B[index].bY - B[index - localSizeY].bY) / device_dx
                     - (B[index].bX - B[index - 1].bX) / device_dy)) * dt;
    }
}

void FieldSolver::timeEvolutionE(
    thrust::device_vector<ElectricField>& E, 
    const thrust::device_vector<MagneticField>& B, 
    const thrust::device_vector<CurrentField>& current, 
    const float dt
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (mPIInfo.localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    timeEvolutionE_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(current.data()), 
        dt, 
        mPIInfo.localSizeX, mPIInfo.localSizeY
    );
    cudaDeviceSynchronize();
}


