#include <fstream>
#include <iomanip>
#include "pic1D.hpp"


PIC1D::PIC1D(MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo), 

      particlesIon     (mPIInfo.totalNumIonPerProcs), 
      particlesElectron(mPIInfo.totalNumElectronPerProcs), 
      E         (mPIInfo.localNx + 2 * mPIInfo.buffer), 
      B         (mPIInfo.localNx + 2 * mPIInfo.buffer), 
      current   (mPIInfo.localNx + 2 * mPIInfo.buffer), 
      tmpE      (mPIInfo.localNx + 2 * mPIInfo.buffer), 
      tmpB      (mPIInfo.localNx + 2 * mPIInfo.buffer), 
      tmpCurrent(mPIInfo.localNx + 2 * mPIInfo.buffer), 

      host_particlesIon     (mPIInfo.totalNumIonPerProcs), 
      host_particlesElectron(mPIInfo.totalNumElectronPerProcs), 
      host_E      (mPIInfo.localNx + 2 * mPIInfo.buffer), 
      host_B      (mPIInfo.localNx + 2 * mPIInfo.buffer), 
      host_current(mPIInfo.localNx + 2 * mPIInfo.buffer)
{

    cudaMalloc(&device_mPIInfo, sizeof(MPIInfo));
    cudaMemcpy(device_mPIInfo, &mPIInfo, sizeof(MPIInfo), cudaMemcpyHostToDevice);

}


__global__ void getCenterBE_kernel(
    MagneticField* tmpB, ElectricField* tmpE, 
    const MagneticField* B, const ElectricField* E, 
    int localNx
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < localNx) {
        int index = i + 1;

        tmpB[index].bX = B[index].bX;
        tmpB[index].bY = 0.5 * (B[index].bY + B[index - 1].bY);
        tmpB[index].bZ = 0.5 * (B[index].bZ + B[index - 1].bZ);
        tmpE[index].eX = 0.5 * (E[index].eX + E[index - 1].eX);
        tmpE[index].eY = E[index].eY;
        tmpE[index].eZ = E[index].eZ;
    }
}

__global__ void getHalfCurrent_kernel(
    CurrentField* current, const CurrentField* tmpCurrent, 
    int localNx
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < localNx) {
        int index = i + 1;

        current[index].jX = 0.5 * (tmpCurrent[index].jX + tmpCurrent[index + 1].jX);
        current[index].jY = tmpCurrent[index].jY;
        current[index].jZ = tmpCurrent[index].jZ;
    }
}


void PIC1D::oneStep()
{
    fieldSolver.timeEvolutionB(B, E, dt / 2.0, mPIInfo);

    sendrecv_field(B, mPIInfo);
    sendrecv_field(E, mPIInfo);
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((mPIInfo.localNx + threadsPerBlock.x - 1) / threadsPerBlock.x);
    getCenterBE_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpB.data()), 
        thrust::raw_pointer_cast(tmpE.data()), 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data()), 
        mPIInfo.localNx
    );
    cudaDeviceSynchronize();
    sendrecv_field(tmpB, mPIInfo);
    sendrecv_field(tmpE, mPIInfo);

    particlePush.pushVelocity(
        particlesIon, particlesElectron, tmpB, tmpE, dt, mPIInfo
    );

    particlePush.pushPosition(
        particlesIon, particlesElectron, dt / 2.0, mPIInfo
    );
    boundary.periodicBoundaryParticleX(
        particlesIon, particlesElectron, mPIInfo
    );

    currentCalculator.resetCurrent(tmpCurrent);
    currentCalculator.calculateCurrent(
        tmpCurrent, particlesIon, particlesElectron, mPIInfo
    );
    sendrecv_field(tmpCurrent, mPIInfo);
    getHalfCurrent_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(tmpCurrent.data()), 
        mPIInfo.localNx
    );
    sendrecv_field(current, mPIInfo);

    fieldSolver.timeEvolutionB(B, E, dt / 2.0, mPIInfo);

    fieldSolver.timeEvolutionE(E, B, current, dt, mPIInfo);

    particlePush.pushPosition(
        particlesIon, particlesElectron, dt / 2.0, mPIInfo
    );
    boundary.periodicBoundaryParticleX(
        particlesIon, particlesElectron, mPIInfo
    );
}


void PIC1D::saveFields(
    std::string directoryname, 
    std::string filenameWithoutStep, 
    int step
)
{
    host_E = E;
    host_B = B;
    host_current = current;
    std::string filenameB, filenameE, filenameCurrent;
    std::string filenameBEnergy, filenameEEnergy;
    double BEnergy = 0.0, EEnergy = 0.0;

    filenameB = directoryname + "/"
             + filenameWithoutStep + "_B_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";
    filenameE = directoryname + "/"
             + filenameWithoutStep + "_E_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";
    filenameCurrent = directoryname + "/"
             + filenameWithoutStep + "_current_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";
    filenameBEnergy = directoryname + "/"
             + filenameWithoutStep + "_BEnergy_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";
    filenameEEnergy = directoryname + "/"
             + filenameWithoutStep + "_EEnergy_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";


    std::ofstream ofsB(filenameB, std::ios::binary);
    ofsB << std::fixed << std::setprecision(6);
    for (int i = 0; i < mPIInfo.localNx; i++) {
        ofsB.write(reinterpret_cast<const char*>(&host_B[i].bX), sizeof(double));
        ofsB.write(reinterpret_cast<const char*>(&host_B[i].bY), sizeof(double));
        ofsB.write(reinterpret_cast<const char*>(&host_B[i].bZ), sizeof(double));
        BEnergy += host_B[i].bX * host_B[i].bX + host_B[i].bY * host_B[i].bY + host_B[i].bZ * host_B[i].bZ;
    }
    BEnergy *= 0.5 / mu0;

    std::ofstream ofsE(filenameE, std::ios::binary);
    ofsE << std::fixed << std::setprecision(6);
    for (int i = 0; i < mPIInfo.localNx; i++) {
        ofsE.write(reinterpret_cast<const char*>(&host_E[i].eX), sizeof(double));
        ofsE.write(reinterpret_cast<const char*>(&host_E[i].eY), sizeof(double));
        ofsE.write(reinterpret_cast<const char*>(&host_E[i].eZ), sizeof(double));
        EEnergy += host_E[i].eX * host_E[i].eX + host_E[i].eY * host_E[i].eY + host_E[i].eZ * host_E[i].eZ;
    }
    EEnergy *= 0.5 * epsilon0;

    std::ofstream ofsCurrent(filenameCurrent, std::ios::binary);
    ofsCurrent << std::fixed << std::setprecision(6);
    for (int i = 0; i < mPIInfo.localNx; i++) {
        ofsCurrent.write(reinterpret_cast<const char*>(&host_current[i].jX), sizeof(double));
        ofsCurrent.write(reinterpret_cast<const char*>(&host_current[i].jY), sizeof(double));
        ofsCurrent.write(reinterpret_cast<const char*>(&host_current[i].jZ), sizeof(double));
    }

    std::ofstream ofsBEnergy(filenameBEnergy, std::ios::binary);
    ofsBEnergy << std::fixed << std::setprecision(6);
    ofsBEnergy.write(reinterpret_cast<const char*>(&BEnergy), sizeof(double));

    std::ofstream ofsEEnergy(filenameEEnergy, std::ios::binary);
    ofsEEnergy << std::fixed << std::setprecision(6);
    ofsEEnergy.write(reinterpret_cast<const char*>(&EEnergy), sizeof(double));
}


void PIC1D::saveParticle(
    std::string directoryname, 
    std::string filenameWithoutStep, 
    int step
)
{
    host_particlesIon = particlesIon;
    host_particlesElectron = particlesElectron;

    std::string filenameXIon, filenameXElectron;
    std::string filenameVIon, filenameVElectron;
    std::string filenameKineticEnergy;

    filenameXIon = directoryname + "/"
             + filenameWithoutStep + "_x_ion_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";
    filenameXElectron = directoryname + "/"
             + filenameWithoutStep + "_x_electron_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";
    filenameVIon = directoryname + "/"
             + filenameWithoutStep + "_v_ion_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";
    filenameVElectron = directoryname + "/"
             + filenameWithoutStep + "_v_electron_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";
    filenameKineticEnergy = directoryname + "/"
             + filenameWithoutStep + "_KE_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";


    std::ofstream ofsXIon(filenameXIon, std::ios::binary);
    ofsXIon << std::fixed << std::setprecision(6);
    for (int i = 0; i < mPIInfo.existNumIonPerProcs; i++) {
        ofsXIon.write(reinterpret_cast<const char*>(&host_particlesIon[i].x), sizeof(double));
        ofsXIon.write(reinterpret_cast<const char*>(&host_particlesIon[i].y), sizeof(double));
        ofsXIon.write(reinterpret_cast<const char*>(&host_particlesIon[i].z), sizeof(double));
    }

    std::ofstream ofsXElectron(filenameXElectron, std::ios::binary);
    ofsXElectron << std::fixed << std::setprecision(6);
    for (int i = 0; i < mPIInfo.existNumElectronPerProcs; i++) {
        ofsXElectron.write(reinterpret_cast<const char*>(&host_particlesElectron[i].x), sizeof(double));
        ofsXElectron.write(reinterpret_cast<const char*>(&host_particlesElectron[i].y), sizeof(double));
        ofsXElectron.write(reinterpret_cast<const char*>(&host_particlesElectron[i].z), sizeof(double));
    }


    double vx, vy, vz, KineticEnergy = 0.0;

    std::ofstream ofsVIon(filenameVIon, std::ios::binary);
    ofsVIon << std::fixed << std::setprecision(6);
    for (int i = 0; i < mPIInfo.existNumIonPerProcs; i++) {
        vx = host_particlesIon[i].vx;
        vy = host_particlesIon[i].vy;
        vz = host_particlesIon[i].vz;

        ofsVIon.write(reinterpret_cast<const char*>(&vx), sizeof(double));
        ofsVIon.write(reinterpret_cast<const char*>(&vy), sizeof(double));
        ofsVIon.write(reinterpret_cast<const char*>(&vz), sizeof(double));

        KineticEnergy += 0.5 * mIon * (vx * vx + vy * vy + vz * vz);
    }

    std::ofstream ofsVElectron(filenameVElectron, std::ios::binary);
    ofsVElectron << std::fixed << std::setprecision(6);
    for (int i = 0; i < mPIInfo.existNumElectronPerProcs; i++) {
        vx = host_particlesElectron[i].vx;
        vy = host_particlesElectron[i].vy;
        vz = host_particlesElectron[i].vz;

        ofsVElectron.write(reinterpret_cast<const char*>(&vx), sizeof(double));
        ofsVElectron.write(reinterpret_cast<const char*>(&vy), sizeof(double));
        ofsVElectron.write(reinterpret_cast<const char*>(&vz), sizeof(double));
        
        KineticEnergy += 0.5 * mElectron * (vx * vx + vy * vy + vz * vz);
    }

    std::ofstream ofsKineticEnergy(filenameKineticEnergy, std::ios::binary);
    ofsKineticEnergy << std::fixed << std::setprecision(6);
    ofsKineticEnergy.write(reinterpret_cast<const char*>(&KineticEnergy), sizeof(double));
}

