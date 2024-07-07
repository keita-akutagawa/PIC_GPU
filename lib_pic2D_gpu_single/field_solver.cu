#include "field_solver.hpp"


__global__ void timeEvolutionB_kernel(
    MagneticField* B, const ElectricField* E, const float dt
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx - 1 && j < device_ny - 1) {
        B[j + device_ny * i].bX += -(E[j + 1 * device_ny * i].eZ - E[j + device_ny * i].eZ) / device_dy * dt;
        B[j + device_ny * i].bY += (E[j + device_ny * (i + 1)].eZ - E[j + device_ny * i].eZ) / device_dx * dt;
        B[j + device_ny * i].bZ += (-(E[j * device_ny * (i + 1)].eY - E[j + device_ny * i].eY) / device_dx
                                 + (E[j + 1 + device_ny * i].eX - E[j + device_ny * i].eX) / device_dy) * dt;
    }

    if (i == device_nx - 1) {
        B[j + device_ny * i].bX += -(E[j + 1 * device_ny * i].eZ - E[j + device_ny * i].eZ) / device_dy * dt;
        B[j + device_ny * i].bY += (E[j + device_ny * 0].eZ - E[j + device_ny * i].eZ) / device_dx * dt;
        B[j + device_ny * i].bZ += (-(E[j * device_ny * 0].eY - E[j + device_ny * i].eY) / device_dx
                                 + (E[j + 1 + device_ny * i].eX - E[j + device_ny * i].eX) / device_dy) * dt;
    }

    if (j == device_ny - 1) {
        B[j + device_ny * i].bX += -(E[0 * device_ny * i].eZ - E[j + device_ny * i].eZ) / device_dy * dt;
        B[j + device_ny * i].bY += (E[j + device_ny * (i + 1)].eZ - E[j + device_ny * i].eZ) / device_dx * dt;
        B[j + device_ny * i].bZ += (-(E[j * device_ny * (i + 1)].eY - E[j + device_ny * i].eY) / device_dx
                                 + (E[0 + device_ny * i].eX - E[j + device_ny * i].eX) / device_dy) * dt;
    }

    if (i == device_nx - 1 && j == device_ny - 1) {
        B[j + device_ny * i].bX += -(E[0 * device_ny * i].eZ - E[j + device_ny * i].eZ) / device_dy * dt;
        B[j + device_ny * i].bY += (E[j + device_ny * 0].eZ - E[j + device_ny * i].eZ) / device_dx * dt;
        B[j + device_ny * i].bZ += (-(E[j * device_ny * 0].eY - E[j + device_ny * i].eY) / device_dx
                                 + (E[0 + device_ny * i].eX - E[j + device_ny * i].eX) / device_dy) * dt;
    }
}

void FieldSolver::timeEvolutionB(
    thrust::device_vector<MagneticField>& B, 
    const thrust::device_vector<ElectricField>& E, 
    const float dt
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    timeEvolutionB_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data()), 
        dt
    );

    cudaDeviceSynchronize();
}



__global__ void timeEvolutionE_kernel(
    ElectricField* E, const MagneticField* B, const CurrentField* current, const float dt
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < device_nx) {
        E[j + device_ny * i].eX += (-current[j + device_ny * i].jX / device_epsilon0
                                 + device_c * device_c * (B[j + device_ny * i].bZ - B[j - 1 + device_ny * i].bZ) / device_dy) * dt;
        E[j + device_ny * i].eY += (-current[j + device_ny * i].jY / device_epsilon0 
                                 - device_c * device_c * (B[j + device_ny * i].bZ - B[j + device_ny * (i - 1)].bZ) / device_dx) * dt;
        E[j + device_ny * i].eZ += (-current[j + device_ny * i].jZ / device_epsilon0 
                                 + device_c * device_c * ((B[j + device_ny * i].bY - B[j + device_ny * (i - 1)].bY) / device_dx
                                 - (B[j + device_ny * i].bX - B[j - 1 + device_ny * i].bX) / device_dy)) * dt;
    }

    if (i == 0) {
        E[j + device_ny * i].eX += (-current[j + device_ny * i].jX / device_epsilon0
                                 + device_c * device_c * (B[j + device_ny * i].bZ - B[j - 1 + device_ny * i].bZ) / device_dy) * dt;
        E[j + device_ny * i].eY += (-current[j + device_ny * i].jY / device_epsilon0 
                                 - device_c * device_c * (B[j + device_ny * i].bZ - B[j + device_ny * (device_nx - 1)].bZ) / device_dx) * dt;
        E[j + device_ny * i].eZ += (-current[j + device_ny * i].jZ / device_epsilon0 
                                 + device_c * device_c * ((B[j + device_ny * i].bY - B[j + device_ny * (device_nx - 1)].bY) / device_dx
                                 - (B[j + device_ny * i].bX - B[j - 1 + device_ny * i].bX) / device_dy)) * dt;
    }

    if (j == 0) {
        E[j + device_ny * i].eX += (-current[j + device_ny * i].jX / device_epsilon0
                                 + device_c * device_c * (B[j + device_ny * i].bZ - B[device_ny - 1 + device_ny * i].bZ) / device_dy) * dt;
        E[j + device_ny * i].eY += (-current[j + device_ny * i].jY / device_epsilon0 
                                 - device_c * device_c * (B[j + device_ny * i].bZ - B[j + device_ny * (i - 1)].bZ) / device_dx) * dt;
        E[j + device_ny * i].eZ += (-current[j + device_ny * i].jZ / device_epsilon0 
                                 + device_c * device_c * ((B[j + device_ny * i].bY - B[j + device_ny * (i - 1)].bY) / device_dx
                                 - (B[j + device_ny * i].bX - B[device_ny - 1 + device_ny * i].bX) / device_dy)) * dt;
    }

    if (i == 0 && j == 0) {
        E[j + device_ny * i].eX += (-current[j + device_ny * i].jX / device_epsilon0
                                 + device_c * device_c * (B[j + device_ny * i].bZ - B[device_ny - 1 + device_ny * i].bZ) / device_dy) * dt;
        E[j + device_ny * i].eY += (-current[j + device_ny * i].jY / device_epsilon0 
                                 - device_c * device_c * (B[j + device_ny * i].bZ - B[j + device_ny * (device_nx - 1)].bZ) / device_dx) * dt;
        E[j + device_ny * i].eZ += (-current[j + device_ny * i].jZ / device_epsilon0 
                                 + device_c * device_c * ((B[j + device_ny * i].bY - B[j + device_ny * (device_nx - 1)].bY) / device_dx
                                 - (B[j + device_ny * i].bX - B[device_ny - 1 + device_ny * i].bX) / device_dy)) * dt;
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
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    timeEvolutionE_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(current.data()), 
        dt
    );

    cudaDeviceSynchronize();
}


