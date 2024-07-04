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


void FieldSolver::timeEvolutionE(
    thrust::device_vector<ElectricField>& E, 
    const thrust::device_vector<MagneticField>& B, 
    const thrust::device_vector<CurrentField>& current, 
    const double dt
)
{
    for (int i = 1; i < nx; i++) {
        E[0][i] += (-current[0][i] / epsilon0) * dt;
        E[1][i] += (-current[1][i] / epsilon0 
                    - c * c * (B[2][i] - B[2][i-1]) / dx) * dt;
        E[2][i] += (-current[2][i] / epsilon0 
                    + c * c * (B[1][i] - B[1][i-1]) / dx) * dt;
    }
    //周期境界条件
    E[0][0] += (-current[0][0] / epsilon0) * dt;
    E[1][0] += (-current[1][0] / epsilon0 
                - c * c * (B[2][0] - B[2][nx-1]) / dx) * dt;
    E[2][0] += (-current[2][0] / epsilon0 
                + c * c * (B[1][0] - B[1][nx-1]) / dx) * dt;
}


