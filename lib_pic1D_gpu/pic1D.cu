#include <fstream>
#include <iomanip>
#include "pic1D.hpp"


PIC1D::PIC1D()
    : particlesIon(totalNumIon), 
      particlesElectron(totalNumElectron), 
      E(nx), 
      B(nx), 
      current(nx), 
      tmpE(nx), 
      tmpB(nx), 
      tmpCurrent(nx), 

      host_particleIon(totalNumIon), 
      host_particleElectron(totalNumElectron), 
      host_E(nx), 
      host_B(nx), 
      host_current(nx)
{
}


__global__ void getCenterBE_kernel(
    MagneticField* tmpB, ElectricField* tmpE, 
    const MagneticField* B, const ElectricField* E
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (0 < i && i < device_nx) {
        tmpB[i].bX = B[i].bX;
        tmpB[i].bY = 0.5 * (B[i].bY + B[i - 1].bY);
        tmpB[i].bZ = 0.5 * (B[i].bZ + B[i - 1].bZ);
        tmpE[i].eX = 0.5 * (E[i].eX + E[i - 1].eX);
        tmpE[i].eY = E[i].eY;
        tmpE[i].eZ = E[i].eZ;
    }

    if (i == 0) {
        tmpB[i].bX = B[i].bX;
        tmpB[i].bY = 0.5 * (B[i].bY + B[device_nx - 1].bY);
        tmpB[i].bZ = 0.5 * (B[i].bZ + B[device_nx - 1].bZ);
        tmpE[i].eX = 0.5 * (E[i].eX + E[device_nx - 1].eX);
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
        current[i].jX = 0.5 * (tmpCurrent[i].jX + tmpCurrent[i + 1].jX);
        current[i].jY = tmpCurrent[i].jY;
        current[i].jZ = tmpCurrent[i].jZ;
    }

    if (i == device_nx - 1) {
        current[i].jX = 0.5 * (tmpCurrent[i].jX + tmpCurrent[0].jX);
        current[i].jY = tmpCurrent[i].jY;
        current[i].jZ = tmpCurrent[i].jZ;
    }
}


void PIC1D::oneStep()
{
    fieldSolver.timeEvolutionB(B, E, dt/2.0);
    boundary.periodicBoundaryBX(B);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x);
    getCenterBE_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpB.data()), 
        thrust::raw_pointer_cast(tmpE.data()), 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data())
    );
    cudaDeviceSynchronize();
    boundary.periodicBoundaryBX(tmpB);
    boundary.periodicBoundaryEX(tmpE);

    particlePush.pushVelocity(
        particlesIon, particlesElectron, tmpB, tmpE, dt
    );

    particlePush.pushPosition(
        particlesIon, particlesElectron, dt/2.0
    );
    boundary.periodicBoundaryParticleX(
        particlesIon, particlesElectron
    );

    currentCalculator.resetCurrent(tmpCurrent);
    currentCalculator.calculateCurrent(
        tmpCurrent, particlesIon, particlesElectron
    );
    boundary.periodicBoundaryCurrentX(tmpCurrent);
    getHalfCurrent_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(tmpCurrent.data())
    );
    boundary.periodicBoundaryCurrentX(current);

    fieldSolver.timeEvolutionB(B, E, dt/2.0);
    boundary.periodicBoundaryBX(B);

    fieldSolver.timeEvolutionE(E, B, current, dt);
    boundary.periodicBoundaryEX(E);

    particlePush.pushPosition(
        particlesIon, particlesElectron, dt/2.0
    );
    boundary.periodicBoundaryParticleX(
        particlesIon, particlesElectron
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
             + ".bin";
    filenameE = directoryname + "/"
             + filenameWithoutStep + "_E_" + std::to_string(step)
             + ".bin";
    filenameCurrent = directoryname + "/"
             + filenameWithoutStep + "_current_" + std::to_string(step)
             + ".bin";
    filenameBEnergy = directoryname + "/"
             + filenameWithoutStep + "_BEnergy_" + std::to_string(step)
             + ".bin";
    filenameEEnergy = directoryname + "/"
             + filenameWithoutStep + "_EEnergy_" + std::to_string(step)
             + ".bin";


    std::ofstream ofsB(filenameB, std::ios::binary);
    ofsB << std::fixed << std::setprecision(6);
    for (int i = 0; i < nx; i++) {
        ofsB.write(reinterpret_cast<const char*>(&host_B[i].bX), sizeof(double));
        ofsB.write(reinterpret_cast<const char*>(&host_B[i].bY), sizeof(double));
        ofsB.write(reinterpret_cast<const char*>(&host_B[i].bZ), sizeof(double));
        BEnergy += host_B[i].bX * host_B[i].bX + host_B[i].bY * host_B[i].bY + host_B[i].bZ * host_B[i].bZ;
    }
    BEnergy *= 0.5 / mu0;

    std::ofstream ofsE(filenameE, std::ios::binary);
    ofsE << std::fixed << std::setprecision(6);
    for (int i = 0; i < nx; i++) {
        ofsE.write(reinterpret_cast<const char*>(&host_E[i].eX), sizeof(double));
        ofsE.write(reinterpret_cast<const char*>(&host_E[i].eY), sizeof(double));
        ofsE.write(reinterpret_cast<const char*>(&host_E[i].eZ), sizeof(double));
        EEnergy += host_E[i].eX * host_E[i].eX + host_E[i].eY * host_E[i].eY + host_E[i].eZ * host_E[i].eZ;
    }
    EEnergy *= 0.5 * epsilon0;

    std::ofstream ofsCurrent(filenameCurrent, std::ios::binary);
    ofsCurrent << std::fixed << std::setprecision(6);
    for (int i = 0; i < nx; i++) {
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
    host_particleIon = particlesIon;
    host_particleElectron = particlesElectron;

    std::string filenameXIon, filenameXElectron;
    std::string filenameVIon, filenameVElectron;
    std::string filenameKineticEnergy;

    filenameXIon = directoryname + "/"
             + filenameWithoutStep + "_x_ion_" + std::to_string(step)
             + ".bin";
    filenameXElectron = directoryname + "/"
             + filenameWithoutStep + "_x_electron_" + std::to_string(step)
             + ".bin";
    filenameVIon = directoryname + "/"
             + filenameWithoutStep + "_v_ion_" + std::to_string(step)
             + ".bin";
    filenameVElectron = directoryname + "/"
             + filenameWithoutStep + "_v_electron_" + std::to_string(step)
             + ".bin";
    filenameKineticEnergy = directoryname + "/"
             + filenameWithoutStep + "_KE_" + std::to_string(step)
             + ".bin";


    std::ofstream ofsXIon(filenameXIon, std::ios::binary);
    ofsXIon << std::fixed << std::setprecision(6);
    for (int i = 0; i < totalNumIon; i++) {
        ofsXIon.write(reinterpret_cast<const char*>(&host_particleIon[i].x), sizeof(double));
        ofsXIon.write(reinterpret_cast<const char*>(&host_particleIon[i].y), sizeof(double));
        ofsXIon.write(reinterpret_cast<const char*>(&host_particleIon[i].z), sizeof(double));
    }

    std::ofstream ofsXElectron(filenameXElectron, std::ios::binary);
    ofsXElectron << std::fixed << std::setprecision(6);
    for (int i = 0; i < totalNumElectron; i++) {
        ofsXElectron.write(reinterpret_cast<const char*>(&host_particleElectron[i].x), sizeof(double));
        ofsXElectron.write(reinterpret_cast<const char*>(&host_particleElectron[i].y), sizeof(double));
        ofsXElectron.write(reinterpret_cast<const char*>(&host_particleElectron[i].z), sizeof(double));
    }


    double vx, vy, vz, KineticEnergy = 0.0;

    std::ofstream ofsVIon(filenameVIon, std::ios::binary);
    ofsVIon << std::fixed << std::setprecision(6);
    for (int i = 0; i < totalNumIon; i++) {
        vx = host_particleIon[i].vx;
        vy = host_particleIon[i].vy;
        vz = host_particleIon[i].vz;

        ofsVIon.write(reinterpret_cast<const char*>(&vx), sizeof(double));
        ofsVIon.write(reinterpret_cast<const char*>(&vy), sizeof(double));
        ofsVIon.write(reinterpret_cast<const char*>(&vz), sizeof(double));

        KineticEnergy += 0.5 * mIon * (vx * vx + vy * vy + vz * vz);
    }

    std::ofstream ofsVElectron(filenameVElectron, std::ios::binary);
    ofsVElectron << std::fixed << std::setprecision(6);
    for (int i = 0; i < totalNumElectron; i++) {
        vx = host_particleElectron[i].vx;
        vy = host_particleElectron[i].vy;
        vz = host_particleElectron[i].vz;

        ofsVElectron.write(reinterpret_cast<const char*>(&vx), sizeof(double));
        ofsVElectron.write(reinterpret_cast<const char*>(&vy), sizeof(double));
        ofsVElectron.write(reinterpret_cast<const char*>(&vz), sizeof(double));
        
        KineticEnergy += 0.5 * mElectron * (vx * vx + vy * vy + vz * vz);
    }

    std::ofstream ofsKineticEnergy(filenameKineticEnergy, std::ios::binary);
    ofsKineticEnergy << std::fixed << std::setprecision(6);
    ofsKineticEnergy.write(reinterpret_cast<const char*>(&KineticEnergy), sizeof(double));
}

