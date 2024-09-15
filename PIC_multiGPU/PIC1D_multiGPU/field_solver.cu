#include "field_solver.hpp"


__global__ void timeEvolutionB_kernel(
    MagneticField* B, const ElectricField* E, 
    const double dt, int localNx
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < localNx) {
        int index = i + 1;

        B[index].bX += 0.0;
        B[index].bY += (E[index + 1].eZ - E[index].eZ) / device_dx * dt;
        B[index].bZ += -(E[index + 1].eY - E[index].eY) / device_dx * dt;
    }
}

void FieldSolver::timeEvolutionB(
    thrust::device_vector<MagneticField>& B, 
    thrust::device_vector<ElectricField>& E, 
    const double dt, 
    MPIInfo& mPIInfo
)
{   
    sendrecv_field(E, mPIInfo);
    MPI_Barrier(MPI_COMM_WORLD);

    int localNx = mPIInfo.localNx;
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((localNx + threadsPerBlock.x - 1) / threadsPerBlock.x);

    timeEvolutionB_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data()), 
        dt, localNx
    );

    cudaDeviceSynchronize();
}



__global__ void timeEvolutionE_kernel(
    ElectricField* E, const MagneticField* B, const CurrentField* current, 
    const double dt, int localNx
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < localNx) {
        int index = i + 1;

        E[index].eX += (-current[index].jX / device_epsilon0) * dt;
        E[index].eY += (-current[index].jY / device_epsilon0 
                 - device_c * device_c * (B[index].bZ - B[index - 1].bZ) / device_dx) * dt;
        E[index].eZ += (-current[index].jZ / device_epsilon0 
                 + device_c * device_c * (B[index].bY - B[index - 1].bY) / device_dx) * dt;
    }
}

void FieldSolver::timeEvolutionE(
    thrust::device_vector<ElectricField>& E, 
    thrust::device_vector<MagneticField>& B, 
    thrust::device_vector<CurrentField>& current, 
    const double dt, 
    MPIInfo& mPIInfo
)
{
    sendrecv_field(B, mPIInfo);
    sendrecv_field(current, mPIInfo);
    MPI_Barrier(MPI_COMM_WORLD);

    int localNx = mPIInfo.localNx;
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((localNx + threadsPerBlock.x - 1) / threadsPerBlock.x);

    timeEvolutionE_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(current.data()), 
        dt, localNx
    );

    cudaDeviceSynchronize();
}


