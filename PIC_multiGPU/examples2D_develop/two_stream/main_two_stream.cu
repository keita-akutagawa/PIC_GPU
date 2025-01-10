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
            B[index].bX = 0.0f;
            B[index].bY = 0.0f;
            B[index].bZ = 0.0f;
        }
    }
}

void PIC2D::initialize()
{
    initializeParticle.uniformForPosition_x(
        0, mPIInfo.existNumIonPerProcs, 
        mPIInfo.xminForProcs, mPIInfo.xmaxForProcs, 
        0, particlesIon
    );
    initializeParticle.uniformForPosition_x(
        0, mPIInfo.existNumElectronPerProcs, 
        mPIInfo.xminForProcs, mPIInfo.xmaxForProcs, 
        100, particlesElectron
    );
    initializeParticle.uniformForPosition_y(
        0, mPIInfo.existNumIonPerProcs, 
        mPIInfo.yminForProcs, mPIInfo.ymaxForProcs, 
        200, particlesIon
    );
    initializeParticle.uniformForPosition_y(
        0, mPIInfo.existNumElectronPerProcs, 
        mPIInfo.yminForProcs, mPIInfo.ymaxForProcs, 
        300, particlesElectron
    );

    initializeParticle.maxwellDistributionForVelocity(
        0.0f, 0.0f, 0.0f, 
        vThIon, vThIon, vThIon, 
        0, mPIInfo.existNumIonPerProcs, 400, particlesIon
    );
    initializeParticle.maxwellDistributionForVelocity(
        -10.0f * vThIon, 0.0f, 0.0f, 
        vThElectron, vThElectron, vThElectron, 
        0, mPIInfo.existNumElectronPerProcs / 2, 500, particlesElectron
    );
    initializeParticle.maxwellDistributionForVelocity(
        +10.0f * vThIon, 0.0f, 0.0f,  
        vThElectron, vThElectron, vThElectron, 
        mPIInfo.existNumElectronPerProcs / 2, mPIInfo.existNumElectronPerProcs, 600, particlesElectron
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

    boundary.periodicBoundaryB_x(B);
    boundary.periodicBoundaryB_y(B);
    boundary.periodicBoundaryE_x(E);
    boundary.periodicBoundaryE_y(E);
    boundary.periodicBoundaryCurrent_x(current);
    boundary.periodicBoundaryCurrent_y(current);
    boundary.boundaryForInitializeParticle_xy(particlesIon, particlesElectron);
    MPI_Barrier(MPI_COMM_WORLD);
}


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    MPIInfo mPIInfo;
    int mpiParticlesNum = 1000000;
    setupInfo(mPIInfo, buffer, mpiParticlesNum);

    if (mPIInfo.rank == 0) {
        std::cout << mPIInfo.gridX << "," << mPIInfo.gridY << std::endl;
        mpifile   << mPIInfo.gridX << "," << mPIInfo.gridY << std::endl;
    }

    cudaSetDevice(mPIInfo.rank);

    initializeDeviceConstants();

    mPIInfo.existNumIonPerProcs      = static_cast<unsigned long long>(totalNumIon / mPIInfo.procs);
    mPIInfo.existNumElectronPerProcs = static_cast<unsigned long long>(totalNumElectron / mPIInfo.procs);
    mPIInfo.totalNumIonPerProcs = mPIInfo.existNumIonPerProcs
                                + numberDensityIon * (mPIInfo.localSizeX + mPIInfo.localSizeY) * (2 * mPIInfo.buffer + 10);
    mPIInfo.totalNumElectronPerProcs = mPIInfo.existNumElectronPerProcs
                                     + numberDensityElectron * (mPIInfo.localSizeX + mPIInfo.localSizeY) * (2 * mPIInfo.buffer + 10);

    mPIInfo.xminForProcs = xmin + (xmax - xmin) / mPIInfo.gridX * mPIInfo.localGridX;
    mPIInfo.xmaxForProcs = xmin + (xmax - xmin) / mPIInfo.gridX * (mPIInfo.localGridX + 1);
    mPIInfo.yminForProcs = ymin + (ymax - ymin) / mPIInfo.gridY * mPIInfo.localGridY;
    mPIInfo.ymaxForProcs = ymin + (ymax - ymin) / mPIInfo.gridY * (mPIInfo.localGridY + 1);

    PIC2D pIC2D(mPIInfo);

    pIC2D.initialize();

    if (mPIInfo.rank == 0) {
        size_t free_mem = 0;
        size_t total_mem = 0;
        cudaError_t status = cudaMemGetInfo(&free_mem, &total_mem);

        std::cout << "Free memory: " << free_mem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Total memory: " << total_mem / (1024 * 1024) << " MB" << std::endl;

        std::cout << "exist number of partices is " 
                  << mPIInfo.procs * (mPIInfo.existNumIonPerProcs + mPIInfo.existNumElectronPerProcs) 
                  << std::endl;
        std::cout << "exist number of partices + buffer particles is " 
                  << mPIInfo.procs * (mPIInfo.totalNumIonPerProcs + mPIInfo.totalNumElectronPerProcs) 
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
        
        pIC2D.oneStep_periodicXY();

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



