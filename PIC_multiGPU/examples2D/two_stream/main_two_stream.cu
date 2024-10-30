#include "main_two_stream_const.hpp"


__global__ void initializeField_kernel(
    ElectricField* E, MagneticField* B, MPIInfo* device_mPIInfo
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx && j < device_ny) {
        MPIInfo mPIInfo = *device_mPIInfo;

        if (mPIInfo.isInside(i, j)) {
            int index = mPIInfo.globalToLocal(i, j);
            E[index].eX = 0.0f;
            E[index].eY = 0.0f;
            E[index].eZ = 0.0f;
            B[index].bX = device_B0;
            B[index].bY = 0.0f;
            B[index].bZ = 0.0f;
        }
    }
}

void PIC2D::initialize()
{
    initializeParticle.uniformForPositionX(
        0, existNumIonPerProcs, 
        xminForProcs, xmaxForProcs, 
        0, particlesIon
    );
    initializeParticle.uniformForPositionX(
        0, existNumElectronPerProcs, 
        xminForProcs, xmaxForProcs, 
        100, particlesElectron
    );
    initializeParticle.uniformForPositionY(
        0, existNumIonPerProcs, 
        yminForProcs, ymaxForProcs, 
        200, particlesIon
    );
    initializeParticle.uniformForPositionY(
        0, existNumElectronPerProcs, 
        yminForProcs, ymaxForProcs, 
        300, particlesElectron
    );

    initializeParticle.maxwellDistributionForVelocity(
        0.0f, 0.0f, 0.0f, 
        vThIon, vThIon, vThIon, 
        0, existNumIonPerProcs, 400, particlesIon
    );
    initializeParticle.maxwellDistributionForVelocity(
        -10.0f * vThIon, 0.0f, 0.0f, 
        vThElectron, vThElectron, vThElectron, 
        0, existNumElectronPerProcs / 2, 500, particlesElectron
    );
    initializeParticle.maxwellDistributionForVelocity(
        +10.0f * vThIon, 0.0f, 0.0f,  
        vThElectron, vThElectron, vThElectron, 
        existNumElectronPerProcs / 2, existNumElectronPerProcs, 600, particlesElectron
    );

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializeField_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), thrust::raw_pointer_cast(B.data()), 
        device_mPIInfo
    );
    cudaDeviceSynchronize();

    MPI_Barrier(MPI_COMM_WORLD);

    sendrecv_field(B, mPIInfo);
    sendrecv_field(E, mPIInfo);
    sendrecv_field(current, mPIInfo);

    boundary.periodicBoundaryBX(B);
    boundary.periodicBoundaryBY(B);
    boundary.periodicBoundaryEX(E);
    boundary.periodicBoundaryEY(E);
    boundary.periodicBoundaryCurrentX(current);
    boundary.periodicBoundaryCurrentY(current);
    boundary.boundaryForInitialize(particlesIon, existNumIonPerProcs);
    boundary.boundaryForInitialize(particlesElectron, existNumElectronPerProcs);
    
    MPI_Barrier(MPI_COMM_WORLD);
}


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    MPIInfo mPIInfo;
    setupInfo(mPIInfo, buffer);

    if (mPIInfo.rank == 0) {
        std::cout << mPIInfo.gridX << "," << mPIInfo.gridY << std::endl;
        mpifile   << mPIInfo.gridX << "," << mPIInfo.gridY << std::endl;
    }

    cudaSetDevice(mPIInfo.rank);

    initializeDeviceConstants();

    existNumIonPerProcs = int(totalNumIon / mPIInfo.procs);
    existNumElectronPerProcs = int(totalNumElectron / mPIInfo.procs);
    totalNumIonPerProcs = existNumIonPerProcs
                                + numberDensityIon * (mPIInfo.localSizeX + mPIInfo.localSizeY) * (2 * mPIInfo.buffer + 10);
    totalNumElectronPerProcs = existNumElectronPerProcs
                                     + numberDensityElectron * (mPIInfo.localSizeX + mPIInfo.localSizeY) * (2 * mPIInfo.buffer + 10);

    xminForProcs = xmin + (xmax - xmin) / mPIInfo.gridX * mPIInfo.localGridX;
    xmaxForProcs = xmin + (xmax - xmin) / mPIInfo.gridX * (mPIInfo.localGridX + 1);
    yminForProcs = ymin + (ymax - ymin) / mPIInfo.gridY * mPIInfo.localGridY;
    ymaxForProcs = ymin + (ymax - ymin) / mPIInfo.gridY * (mPIInfo.localGridY + 1);

    PIC2D pIC2D(mPIInfo);

    pIC2D.initialize();

    if (mPIInfo.rank == 0) {
        size_t free_mem = 0;
        size_t total_mem = 0;
        cudaError_t status = cudaMemGetInfo(&free_mem, &total_mem);

        std::cout << "Free memory: " << free_mem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Total memory: " << total_mem / (1024 * 1024) << " MB" << std::endl;

        std::cout << "exist number of partices is " 
                  << mPIInfo.procs * (existNumIonPerProcs + existNumElectronPerProcs) 
                  << std::endl;
        std::cout << "exist number of partices + buffer particles is " 
                  << mPIInfo.procs * (totalNumIonPerProcs + totalNumElectronPerProcs) 
                  << std::endl;
        std::cout << std::setprecision(4) 
                << "omega_pe * t = " << totalStep * dt * omegaPe << std::endl;
    }


    for (int step = 0; step < totalStep + 1; step++) {
        MPI_Barrier(MPI_COMM_WORLD);

        if (step % fieldRecordStep == 0) {
            if (mPIInfo.rank == 0) {
                std::cout << std::to_string(step) << " step done : total time is "
                            << std::setprecision(6) << totalTime
                            << std::endl;
                logfile << std::to_string(step) << "," << totalTime << std::endl;
            }
            pIC2D.saveFields(
                directoryname, filenameWithoutStep, step
            );
            pIC2D.saveParticle(
                directoryname, filenameWithoutStep, step
            );
        }
        
        pIC2D.oneStepPeriodicXY();

        if (mPIInfo.rank == 0) {
            totalTime += dt;
        }
    }

    MPI_Finalize();

    if (mPIInfo.rank == 0) {
        std::cout << "program was completed!" << std::endl;
    }

    return 0;
}



