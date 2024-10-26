#include "main_weibel_const.hpp"


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
            E[index].eX = 0.0;
            E[index].eY = 0.0;
            E[index].eZ = 0.0;
            B[index].bX = 0.0;
            B[index].bY = 0.0;
            B[index].bZ = 0.0;
        }
    }
}

void PIC2D::initialize()
{
    float xminForProcs = xmin + (xmax - xmin) / mPIInfo.gridX * mPIInfo.localGridX;
    float xmaxForProcs = xmin + (xmax - xmin) / mPIInfo.gridX * (mPIInfo.localGridX + 1);
    float yminForProcs = ymin + (ymax - ymin) / mPIInfo.gridY * mPIInfo.localGridY;
    float ymaxForProcs = ymin + (ymax - ymin) / mPIInfo.gridY * (mPIInfo.localGridY + 1);

    initializeParticle.uniformForPositionX(
        0, mPIInfo.existNumIonPerProcs, 
        xminForProcs, xmaxForProcs, 
        0, particlesIon
    );
    initializeParticle.uniformForPositionX(
        0, mPIInfo.existNumElectronPerProcs, 
        xminForProcs, xmaxForProcs, 
        100, particlesElectron
    );
    initializeParticle.uniformForPositionY(
        0, mPIInfo.existNumIonPerProcs, 
        yminForProcs, ymaxForProcs, 
        200, particlesIon
    );
    initializeParticle.uniformForPositionY(
        0, mPIInfo.existNumElectronPerProcs, 
        yminForProcs, ymaxForProcs, 
        300, particlesElectron
    );

    initializeParticle.maxwellDistributionForVelocity(
        bulkVxIon, bulkVyIon, bulkVzIon, 
        vThIon, vThIon, 5.0 * vThIon, 
        0, mPIInfo.existNumIonPerProcs, 400, particlesIon
    );
    initializeParticle.maxwellDistributionForVelocity(
        bulkVxElectron, bulkVyElectron, bulkVzElectron, 
        vThElectron, vThElectron, 5.0 * vThElectron, 
        0, mPIInfo.existNumElectronPerProcs, 500, particlesElectron
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
    boundary.periodicBoundaryParticleXY(particlesIon, particlesElectron);

    MPI_Barrier(MPI_COMM_WORLD);
}


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    MPIInfo mPIInfo;
    setupInfo(mPIInfo, buffer);

    cudaSetDevice(mPIInfo.rank);

    initializeDeviceConstants();

    mPIInfo.existNumIonPerProcs = int(totalNumIon / mPIInfo.procs);
    mPIInfo.existNumElectronPerProcs = int(totalNumElectron / mPIInfo.procs);
    mPIInfo.totalNumIonPerProcs = mPIInfo.existNumIonPerProcs + 100000;
    mPIInfo.totalNumElectronPerProcs = mPIInfo.existNumElectronPerProcs + 100000;

    PIC2D pIC2D(mPIInfo);

    pIC2D.initialize();

    if (mPIInfo.rank == 0) {
         size_t free_mem = 0;
        size_t total_mem = 0;
        cudaError_t status = cudaMemGetInfo(&free_mem, &total_mem);

        std::cout << "Free memory: " << free_mem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Total memory: " << total_mem / (1024 * 1024) << " MB" << std::endl;

        std::cout << "total number of partices is " << totalNumParticles << std::endl;
        std::cout << std::setprecision(4) 
                << "omega_pe * t = " << totalStep * dt * omegaPe << std::endl;
    }


    for (int step = 0; step < totalStep + 1; step++) {
        MPI_Barrier(MPI_COMM_WORLD);

        if (step % fieldRecordStep == 0) {
            if (mPIInfo.rank == 0) {
                std::cout << std::to_string(step) << " step done : total time is "
                            << std::setprecision(4) << totalTime
                            << std::endl;
                logfile << std::setprecision(6) << totalTime << std::endl;
            }
            pIC2D.saveFields(
                directoryname, filenameWithoutStep, step
            );
            pIC2D.saveZerothMoments(
                directoryname, filenameWithoutStep, step
            );
            pIC2D.saveFirstMoments(
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



