#include "main_two_stream_electron_const.hpp"


__global__ void initializeField_kernel(
    ElectricField* E, MagneticField* B, int nx, MPIInfo* device_mPIInfo
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nx) {
        MPIInfo mPIInfo = *device_mPIInfo;

        if (mPIInfo.isInside(i)) {
            int index = mPIInfo.globalToLocal(i);
            E[index].eX = 0.0f;
            E[index].eY = 0.0f;
            E[index].eZ = 0.0f;
            B[index].bX = device_B0;
            B[index].bY = 0.0f;
            B[index].bZ = 0.0f;
        }
    }
}

void PIC1D::initialize()
{
    initializeParticle.uniformForPositionX(
        0, mPIInfo.existNumIonPerProcs, 
        xmin + (xmax - xmin) / mPIInfo.procs * mPIInfo.rank, xmin + (xmax - xmin) / mPIInfo.procs * (mPIInfo.rank + 1), 
        0 + mPIInfo.rank, particlesIon
    );
    initializeParticle.uniformForPositionX(
        0, mPIInfo.existNumElectronPerProcs, 
        xmin + (xmax - xmin) / mPIInfo.procs * mPIInfo.rank, xmin + (xmax - xmin) / mPIInfo.procs * (mPIInfo.rank + 1), 
        100000 + mPIInfo.rank, particlesElectron
    );

    initializeParticle.maxwellDistributionForVelocity(
        bulkVxIon, bulkVyIon, bulkVzIon, vThIon, 
        0, mPIInfo.existNumIonPerProcs, 
        200000 + mPIInfo.rank, particlesIon
    );
    initializeParticle.maxwellDistributionForVelocity(
        bulkVxElectron, bulkVyElectron, bulkVzElectron, vThElectron, 
        0, mPIInfo.existNumElectronPerProcs / 2, 
        300000 + mPIInfo.rank, particlesElectron
    );
    initializeParticle.maxwellDistributionForVelocity(
        bulkVxElectronBeam, bulkVyElectronBeam, bulkVzElectronBeam, vThElectron, 
        mPIInfo.existNumElectronPerProcs / 2, mPIInfo.existNumElectronPerProcs, 
        400000 + mPIInfo.rank, particlesElectron
    );


    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x);

    initializeField_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), thrust::raw_pointer_cast(B.data()), 
        nx, device_mPIInfo
    );

    cudaDeviceSynchronize();
}


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    MPIInfo mPIInfo;
    setupInfo(mPIInfo);

    cudaSetDevice(mPIInfo.rank);

    initializeDeviceConstants();

    cudaMemcpyToSymbol(device_totalNumElectronBeam1, &totalNumElectronBeam1, sizeof(int));
    cudaMemcpyToSymbol(device_totalNumElectronBeam2, &totalNumElectronBeam2, sizeof(int));
    cudaMemcpyToSymbol(device_bulkVxElectronBeam, &bulkVxElectronBeam, sizeof(float));
    cudaMemcpyToSymbol(device_bulkVxElectronBeam, &bulkVxElectronBeam, sizeof(float));
    cudaMemcpyToSymbol(device_bulkVxElectronBeam, &bulkVxElectronBeam, sizeof(float));

    mPIInfo.existNumIonPerProcs = int(totalNumIon / mPIInfo.procs);
    mPIInfo.existNumElectronPerProcs = int(totalNumElectron / mPIInfo.procs);
    mPIInfo.totalNumIonPerProcs = totalNumIon;
    mPIInfo.totalNumElectronPerProcs = totalNumElectron;

    
    PIC1D pIC1D(mPIInfo);

    pIC1D.initialize();

    
    if (mPIInfo.rank == 0) {
        std::cout << "total number of partices is " << totalNumParticles << std::endl;
        std::cout << std::setprecision(4) 
                  << "omega_pe * t = " << totalStep * dt * omegaPe << std::endl;
    }

    for (int step = 0; step < 1; step++) {
        if (step % recordStep == 0) {
            if (mPIInfo.rank == 0) {
                std::cout << std::to_string(step) << " step done : total time is "
                            << std::setprecision(4) << totalTime
                            << std::endl;
                logfile << std::setprecision(6) << totalTime << std::endl;
            }
            pIC1D.saveFields(
                directoryname, filenameWithoutStep, step
            );
            pIC1D.saveParticle(
                directoryname, filenameWithoutStep, step
            );
        }

        pIC1D.oneStep();

        if (mPIInfo.rank == 0) {
            totalTime += dt;
        }
    }


    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    if (mPIInfo.rank == 0) {
        std::cout << "program was completed!" << std::endl;
    }

    return 0;
}


