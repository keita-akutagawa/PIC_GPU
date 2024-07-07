#include "field_solver.hpp"


__global__ void timeEvolutionB_kernel(
    MagneticField* B, const ElectricField* E, const double dt
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < device_nx - 1) {
        B[i].bX += 0.0;
        B[i].bY += (E[i + 1].eZ - E[i].eZ) / device_dx * dt;
        B[i].bZ += -(E[i + 1].eY - E[i].eY) / device_dx * dt;
    }

    if (i == device_nx - 1) {
        B[i].bX += 0.0;
        B[i].bY += (E[0].eZ - E[i].eZ) / device_dx * dt;
        B[i].bZ += -(E[0].eY - E[i].eY) / device_dx * dt;
    }
}

void FieldSolver::timeEvolutionB(
    thrust::device_vector<MagneticField>& B, 
    const thrust::device_vector<ElectricField>& E, 
    const double dt
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x);

    timeEvolutionB_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data()), 
        dt
    );

    cudaDeviceSynchronize();
}



__global__ void timeEvolutionE_kernel(
    ElectricField* E, const MagneticField* B, const CurrentField* current, const double dt
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (0 < i && i < device_nx) {
        E[i].eX += (-current[i].jX / device_epsilon0) * dt;
        E[i].eY += (-current[i].jY / device_epsilon0 
                 - device_c * device_c * (B[i].bZ - B[i - 1].bZ) / device_dx) * dt;
        E[i].eZ += (-current[i].jZ / device_epsilon0 
                 + device_c * device_c * (B[i].bY - B[i - 1].bY) / device_dx) * dt;
    }

    if (i == 0) {
        E[i].eX += (-current[i].jX / device_epsilon0) * dt;
        E[i].eY += (-current[i].jY / device_epsilon0 
                 - device_c * device_c * (B[i].bZ - B[device_nx - 1].bZ) / device_dx) * dt;
        E[i].eZ += (-current[i].jZ / device_epsilon0 
                 + device_c * device_c * (B[i].bY - B[device_nx - 1].bY) / device_dx) * dt;
    }
}

void FieldSolver::timeEvolutionE(
    thrust::device_vector<ElectricField>& E, 
    const thrust::device_vector<MagneticField>& B, 
    const thrust::device_vector<CurrentField>& current, 
    const double dt
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x);

    timeEvolutionE_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(current.data()), 
        dt
    );

    cudaDeviceSynchronize();
}


