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
    const MagneticField* B, const ElectricField* E
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (0 < i && i < device_nx) {
        tmpB[i].bX = B[i].bX;
        tmpB[i].bY = 0.5f * (B[i].bY + B[i - 1].bY);
        tmpB[i].bZ = 0.5f * (B[i].bZ + B[i - 1].bZ);
        tmpE[i].eX = 0.5f * (E[i].eX + E[i - 1].eX);
        tmpE[i].eY = E[i].eY;
        tmpE[i].eZ = E[i].eZ;
    }

    if (i == 0) {
        tmpB[i].bX = B[i].bX;
        tmpB[i].bY = 0.5f * (B[i].bY + B[device_nx - 1].bY);
        tmpB[i].bZ = 0.5f * (B[i].bZ + B[device_nx - 1].bZ);
        tmpE[i].eX = 0.5f * (E[i].eX + E[device_nx - 1].eX);
        tmpE[i].eY = E[i].eY;
        tmpE[i].eZ = E[i].eZ;
    }
}

__global__ void getHalfCurrent_kernel(
    CurrentField* current, const CurrentField* tmpCurrent
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < device_nx - 1) {
        current[i].jX = 0.5f * (tmpCurrent[i].jX + tmpCurrent[i + 1].jX);
        current[i].jY = tmpCurrent[i].jY;
        current[i].jZ = tmpCurrent[i].jZ;
    }

    if (i == device_nx - 1) {
        current[i].jX = 0.5f * (tmpCurrent[i].jX + tmpCurrent[0].jX);
        current[i].jY = tmpCurrent[i].jY;
        current[i].jZ = tmpCurrent[i].jZ;
    }
}


void PIC1D::oneStep()
{
    fieldSolver.timeEvolutionB(B, E, dt/2.0, mPIInfo);
    boundary.periodicBoundaryBX(B);

    //dim3 threadsPerBlock(256);
    //dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x);
    //getCenterBE_kernel<<<blocksPerGrid, threadsPerBlock>>>(
    //    thrust::raw_pointer_cast(tmpB.data()), 
    //    thrust::raw_pointer_cast(tmpE.data()), 
    //    thrust::raw_pointer_cast(B.data()), 
    //    thrust::raw_pointer_cast(E.data())
    //);
    //cudaDeviceSynchronize();
    //boundary.periodicBoundaryBX(tmpB);
    //boundary.periodicBoundaryEX(tmpE);

    //particlePush.pushVelocity(
    //    particlesIon, particlesElectron, tmpB, tmpE, dt
    //);

    //particlePush.pushPosition(
    //    particlesIon, particlesElectron, dt/2.0f
    //);
    //boundary.periodicBoundaryParticleX(
    //    particlesIon, particlesElectron
    //);

    //currentCalculator.resetCurrent(tmpCurrent);
    //currentCalculator.calculateCurrent(
    //    tmpCurrent, particlesIon, particlesElectron
    //);
    //boundary.periodicBoundaryCurrentX(tmpCurrent);
    //getHalfCurrent_kernel<<<blocksPerGrid, threadsPerBlock>>>(
    //    thrust::raw_pointer_cast(current.data()), 
    //    thrust::raw_pointer_cast(tmpCurrent.data())
    //);
    //boundary.periodicBoundaryCurrentX(current);

    fieldSolver.timeEvolutionB(B, E, dt/2.0f, mPIInfo);
    boundary.periodicBoundaryBX(B);

    fieldSolver.timeEvolutionE(E, B, current, dt, mPIInfo);
    boundary.periodicBoundaryEX(E);

    //particlePush.pushPosition(
    //    particlesIon, particlesElectron, dt/2.0f
    //);
    //boundary.periodicBoundaryParticleX(
    //    particlesIon, particlesElectron
    //);
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
    float BEnergy = 0.0f, EEnergy = 0.0f;

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
        ofsB.write(reinterpret_cast<const char*>(&host_B[i].bX), sizeof(float));
        ofsB.write(reinterpret_cast<const char*>(&host_B[i].bY), sizeof(float));
        ofsB.write(reinterpret_cast<const char*>(&host_B[i].bZ), sizeof(float));
        BEnergy += host_B[i].bX * host_B[i].bX + host_B[i].bY * host_B[i].bY + host_B[i].bZ * host_B[i].bZ;
    }
    BEnergy *= 0.5f / mu0;

    std::ofstream ofsE(filenameE, std::ios::binary);
    ofsE << std::fixed << std::setprecision(6);
    for (int i = 0; i < mPIInfo.localNx; i++) {
        ofsE.write(reinterpret_cast<const char*>(&host_E[i].eX), sizeof(float));
        ofsE.write(reinterpret_cast<const char*>(&host_E[i].eY), sizeof(float));
        ofsE.write(reinterpret_cast<const char*>(&host_E[i].eZ), sizeof(float));
        EEnergy += host_E[i].eX * host_E[i].eX + host_E[i].eY * host_E[i].eY + host_E[i].eZ * host_E[i].eZ;
    }
    EEnergy *= 0.5f * epsilon0;

    std::ofstream ofsCurrent(filenameCurrent, std::ios::binary);
    ofsCurrent << std::fixed << std::setprecision(6);
    for (int i = 0; i < mPIInfo.localNx; i++) {
        ofsCurrent.write(reinterpret_cast<const char*>(&host_current[i].jX), sizeof(float));
        ofsCurrent.write(reinterpret_cast<const char*>(&host_current[i].jY), sizeof(float));
        ofsCurrent.write(reinterpret_cast<const char*>(&host_current[i].jZ), sizeof(float));
    }

    std::ofstream ofsBEnergy(filenameBEnergy, std::ios::binary);
    ofsBEnergy << std::fixed << std::setprecision(6);
    ofsBEnergy.write(reinterpret_cast<const char*>(&BEnergy), sizeof(float));

    std::ofstream ofsEEnergy(filenameEEnergy, std::ios::binary);
    ofsEEnergy << std::fixed << std::setprecision(6);
    ofsEEnergy.write(reinterpret_cast<const char*>(&EEnergy), sizeof(float));
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
        ofsXIon.write(reinterpret_cast<const char*>(&host_particlesIon[i].x), sizeof(float));
        ofsXIon.write(reinterpret_cast<const char*>(&host_particlesIon[i].y), sizeof(float));
        ofsXIon.write(reinterpret_cast<const char*>(&host_particlesIon[i].z), sizeof(float));
    }

    std::ofstream ofsXElectron(filenameXElectron, std::ios::binary);
    ofsXElectron << std::fixed << std::setprecision(6);
    for (int i = 0; i < mPIInfo.existNumElectronPerProcs; i++) {
        ofsXElectron.write(reinterpret_cast<const char*>(&host_particlesElectron[i].x), sizeof(float));
        ofsXElectron.write(reinterpret_cast<const char*>(&host_particlesElectron[i].y), sizeof(float));
        ofsXElectron.write(reinterpret_cast<const char*>(&host_particlesElectron[i].z), sizeof(float));
    }


    float vx, vy, vz, KineticEnergy = 0.0f;

    std::ofstream ofsVIon(filenameVIon, std::ios::binary);
    ofsVIon << std::fixed << std::setprecision(6);
    for (int i = 0; i < mPIInfo.existNumIonPerProcs; i++) {
        vx = host_particlesIon[i].vx;
        vy = host_particlesIon[i].vy;
        vz = host_particlesIon[i].vz;

        ofsVIon.write(reinterpret_cast<const char*>(&vx), sizeof(float));
        ofsVIon.write(reinterpret_cast<const char*>(&vy), sizeof(float));
        ofsVIon.write(reinterpret_cast<const char*>(&vz), sizeof(float));

        KineticEnergy += 0.5f * mIon * (vx * vx + vy * vy + vz * vz);
    }

    std::ofstream ofsVElectron(filenameVElectron, std::ios::binary);
    ofsVElectron << std::fixed << std::setprecision(6);
    for (int i = 0; i < mPIInfo.existNumElectronPerProcs; i++) {
        vx = host_particlesElectron[i].vx;
        vy = host_particlesElectron[i].vy;
        vz = host_particlesElectron[i].vz;

        ofsVElectron.write(reinterpret_cast<const char*>(&vx), sizeof(float));
        ofsVElectron.write(reinterpret_cast<const char*>(&vy), sizeof(float));
        ofsVElectron.write(reinterpret_cast<const char*>(&vz), sizeof(float));
        
        KineticEnergy += 0.5f * mElectron * (vx * vx + vy * vy + vz * vz);
    }

    std::ofstream ofsKineticEnergy(filenameKineticEnergy, std::ios::binary);
    ofsKineticEnergy << std::fixed << std::setprecision(6);
    ofsKineticEnergy.write(reinterpret_cast<const char*>(&KineticEnergy), sizeof(float));
}

