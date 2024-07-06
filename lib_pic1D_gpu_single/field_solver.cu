#include "field_solver.hpp"


__global__ void timeEvolutionB_kernel(
    MagneticField* magneticField, const ElectricField* electricField, const double dt
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < device_nx - 1) {
        magneticField[i].bX += 0.0f;
        magneticField[i].bY += (electricField[i + 1].eZ - electricField[i].eZ) / device_dx * dt;
        magneticField[i].bZ += -(electricField[i + 1].eY - electricField[i].eY) / device_dx * dt;
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
    ElectricField* electricField, const MagneticField* magneticField, const CurrentField* currentField, const double dt
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (0 < i && i < device_nx) {
        electricField[i].eX += (-currentField[i].jX / device_epsilon0) * dt;
        electricField[i].eY += (-currentField[i].jY / device_epsilon0 
                             - device_c * device_c * (magneticField[i].bZ - magneticField[i - 1].bZ) / device_dx) * dt;
        electricField[i].eZ += (-currentField[i].jZ / device_epsilon0 
                             + device_c * device_c * (magneticField[i].bY - magneticField[i - 1].bY) / device_dx) * dt;
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


