#include <fstream>
#include <iomanip>
#include "pic2D.hpp"


PIC2D::PIC2D()
    : particlesIon(totalNumIon), 
      particlesElectron(totalNumElectron), 
      E(nx * ny), 
      B(nx * ny), 
      current(nx * ny), 
      tmpE(nx * ny), 
      tmpB(nx * ny), 
      tmpCurrent(nx * ny), 

      host_particlesIon(totalNumIon), 
      host_particlesElectron(totalNumElectron), 
      host_E(nx * ny), 
      host_B(nx * ny), 
      host_current(nx * ny)
{
}


__global__ void getCenterBE_kernel(
    MagneticField* tmpB, ElectricField* tmpE, 
    const MagneticField* B, const ElectricField* E
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < device_nx) && (0 < j) && (j < device_ny)) {
        tmpB[j + device_ny * i].bX = 0.5f * (B[j + device_ny * i].bX + B[j - 1 + device_ny * i].bX);
        tmpB[j + device_ny * i].bY = 0.5f * (B[j + device_ny * i].bY + B[j + device_ny * (i - 1)].bY);
        tmpB[j + device_ny * i].bZ = 0.25f * (B[j + device_ny * i].bZ + B[j + device_ny * (i - 1)].bZ
                                   + B[j - 1 + device_ny * i].bZ + B[j - 1 + device_ny * (i - 1)].bZ);
        tmpE[j + device_ny * i].eX = 0.5f * (E[j + device_ny * i].eX + E[j + device_ny * (i - 1)].eX);
        tmpE[j + device_ny * i].eY = 0.5f * (E[j + device_ny * i].eY + E[j - 1 + device_ny * i].eY);
        tmpE[j + device_ny * i].eZ = E[j + device_ny * i].eZ;
    }

    if (i == 0) {
        tmpB[j + device_ny * i].bX = 0.5f * (B[j + device_ny * i].bX + B[j - 1 + device_ny * i].bX);
        tmpB[j + device_ny * i].bY = 0.5f * (B[j + device_ny * i].bY + B[j + device_ny * (device_nx - 1)].bY);
        tmpB[j + device_ny * i].bZ = 0.25f * (B[j + device_ny * i].bZ + B[j + device_ny * (device_nx - 1)].bZ
                                   + B[j - 1 + device_ny * i].bZ + B[j - 1 + device_ny * (device_nx - 1)].bZ);
        tmpE[j + device_ny * i].eX = 0.5f * (E[j + device_ny * i].eX + E[j + device_ny * (device_nx - 1)].eX);
        tmpE[j + device_ny * i].eY = 0.5f * (E[j + device_ny * i].eY + E[j - 1 + device_ny * i].eY);
        tmpE[j + device_ny * i].eZ = E[j + device_ny * i].eZ;
    }

    if (j == 0) {
        tmpB[j + device_ny * i].bX = 0.5f * (B[j + device_ny * i].bX + B[device_ny - 1 + device_ny * i].bX);
        tmpB[j + device_ny * i].bY = 0.5f * (B[j + device_ny * i].bY + B[j + device_ny * (i - 1)].bY);
        tmpB[j + device_ny * i].bZ = 0.25f * (B[j + device_ny * i].bZ + B[j + device_ny * (i - 1)].bZ
                                   + B[device_ny - 1 + device_ny * i].bZ + B[device_ny - 1 + device_ny * (i - 1)].bZ);
        tmpE[j + device_ny * i].eX = 0.5f * (E[j + device_ny * i].eX + E[j + device_ny * (i - 1)].eX);
        tmpE[j + device_ny * i].eY = 0.5f * (E[j + device_ny * i].eY + E[device_ny - 1 + device_ny * i].eY);
        tmpE[j + device_ny * i].eZ = E[j + device_ny * i].eZ;
    }

    if (i == 0 && j == 0) {
        tmpB[j + device_ny * i].bX = 0.5f * (B[j + device_ny * i].bX + B[device_ny - 1 + device_ny * i].bX);
        tmpB[j + device_ny * i].bY = 0.5f * (B[j + device_ny * i].bY + B[j + device_ny * (device_nx - 1)].bY);
        tmpB[j + device_ny * i].bZ = 0.25f * (B[j + device_ny * i].bZ + B[j + device_ny * (device_nx - 1)].bZ
                                   + B[device_ny - 1 + device_ny * i].bZ + B[device_ny - 1 + device_ny * (device_nx - 1)].bZ);
        tmpE[j + device_ny * i].eX = 0.5f * (E[j + device_ny * i].eX + E[j + device_ny * (device_nx - 1)].eX);
        tmpE[j + device_ny * i].eY = 0.5f * (E[j + device_ny * i].eY + E[device_ny - 1 + device_ny * i].eY);
        tmpE[j + device_ny * i].eZ = E[j + device_ny * i].eZ;
    }
}

__global__ void getHalfCurrent_kernel(
    CurrentField* current, const CurrentField* tmpCurrent
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx - 1 && j < device_ny - 1) {
        current[j + device_ny * i].jX = 0.5f * (tmpCurrent[j + device_ny * i].jX + tmpCurrent[j + device_ny * (i + 1)].jX);
        current[j + device_ny * i].jY = 0.5f * (tmpCurrent[j + device_ny * i].jY + tmpCurrent[j + 1 + device_ny * i].jY);
        current[j + device_ny * i].jZ = tmpCurrent[i].jZ;
    }

    if (i == device_nx - 1) {
        current[j + device_ny * i].jX = 0.5f * (tmpCurrent[j + device_ny * i].jX + tmpCurrent[j + device_ny * 0].jX);
        current[j + device_ny * i].jY = 0.5f * (tmpCurrent[j + device_ny * i].jY + tmpCurrent[j + 1 + device_ny * i].jY);
        current[j + device_ny * i].jZ = tmpCurrent[i].jZ;
    }

    if (j == device_ny - 1) {
        current[j + device_ny * i].jX = 0.5f * (tmpCurrent[j + device_ny * i].jX + tmpCurrent[j + device_ny * (i + 1)].jX);
        current[j + device_ny * i].jY = 0.5f * (tmpCurrent[j + device_ny * i].jY + tmpCurrent[0 + device_ny * i].jY);
        current[j + device_ny * i].jZ = tmpCurrent[i].jZ;
    }

    if (i == device_nx - 1 && j == device_ny - 1) {
        current[j + device_ny * i].jX = 0.5f * (tmpCurrent[j + device_ny * i].jX + tmpCurrent[j + device_ny * 0].jX);
        current[j + device_ny * i].jY = 0.5f * (tmpCurrent[j + device_ny * i].jY + tmpCurrent[0 + device_ny * i].jY);
        current[j + device_ny * i].jZ = tmpCurrent[i].jZ;
    }
}


void PIC2D::oneStep()
{
    fieldSolver.timeEvolutionB(B, E, dt/2.0);
    boundary.periodicBoundaryBX(B);
    boundary.periodicBoundaryBY(B);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);
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
        particlesIon, particlesElectron, dt/2.0f
    );
    boundary.periodicBoundaryParticleX(
        particlesIon, particlesElectron
    );
    boundary.periodicBoundaryParticleY(
        particlesIon, particlesElectron
    );

    currentCalculator.resetCurrent(tmpCurrent);
    currentCalculator.calculateCurrent(
        tmpCurrent, particlesIon, particlesElectron
    );
    boundary.periodicBoundaryCurrentX(tmpCurrent);
    boundary.periodicBoundaryCurrentY(tmpCurrent);
    getHalfCurrent_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(tmpCurrent.data())
    );
    boundary.periodicBoundaryCurrentX(current);
    boundary.periodicBoundaryCurrentY(current);

    fieldSolver.timeEvolutionB(B, E, dt/2.0f);
    boundary.periodicBoundaryBX(B);
    boundary.periodicBoundaryBY(B);

    fieldSolver.timeEvolutionE(E, B, current, dt);
    boundary.periodicBoundaryEX(E);
    boundary.periodicBoundaryEY(E);

    particlePush.pushPosition(
        particlesIon, particlesElectron, dt/2.0f
    );
    boundary.periodicBoundaryParticleX(
        particlesIon, particlesElectron
    );
    boundary.periodicBoundaryParticleY(
        particlesIon, particlesElectron
    );
}


void PIC2D::saveFields(
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
        for (int j = 0; j < ny; j++) {
            ofsB.write(reinterpret_cast<const char*>(&host_B[j + device_ny * i].bX), sizeof(float));
            ofsB.write(reinterpret_cast<const char*>(&host_B[j + device_ny * i].bY), sizeof(float));
            ofsB.write(reinterpret_cast<const char*>(&host_B[j + device_ny * i].bZ), sizeof(float));
            BEnergy += host_B[j + device_ny * i].bX * host_B[j + device_ny * i].bX 
                     + host_B[j + device_ny * i].bY * host_B[j + device_ny * i].bY
                     + host_B[j + device_ny * i].bZ * host_B[j + device_ny * i].bZ;
        }
    }
    BEnergy *= 0.5f / mu0;

    std::ofstream ofsE(filenameE, std::ios::binary);
    ofsE << std::fixed << std::setprecision(6);
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            ofsE.write(reinterpret_cast<const char*>(&host_E[j + device_ny * i].eX), sizeof(float));
            ofsE.write(reinterpret_cast<const char*>(&host_E[j + device_ny * i].eY), sizeof(float));
            ofsE.write(reinterpret_cast<const char*>(&host_E[j + device_ny * i].eZ), sizeof(float));
            EEnergy += host_E[j + device_ny * i].eX * host_E[j + device_ny * i].eX
                     + host_E[j + device_ny * i].eY * host_E[j + device_ny * i].eY
                     + host_E[j + device_ny * i].eZ * host_E[j + device_ny * i].eZ;
        }
    }
    EEnergy *= 0.5f * epsilon0;

    std::ofstream ofsCurrent(filenameCurrent, std::ios::binary);
    ofsCurrent << std::fixed << std::setprecision(6);
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            ofsCurrent.write(reinterpret_cast<const char*>(&host_current[j + device_ny * i].jX), sizeof(float));
            ofsCurrent.write(reinterpret_cast<const char*>(&host_current[j + device_ny * i].jY), sizeof(float));
            ofsCurrent.write(reinterpret_cast<const char*>(&host_current[j + device_ny * i].jZ), sizeof(float));
        }
    }

    std::ofstream ofsBEnergy(filenameBEnergy, std::ios::binary);
    ofsBEnergy << std::fixed << std::setprecision(6);
    ofsBEnergy.write(reinterpret_cast<const char*>(&BEnergy), sizeof(float));

    std::ofstream ofsEEnergy(filenameEEnergy, std::ios::binary);
    ofsEEnergy << std::fixed << std::setprecision(6);
    ofsEEnergy.write(reinterpret_cast<const char*>(&EEnergy), sizeof(float));
}


void PIC2D::saveParticle(
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
        ofsXIon.write(reinterpret_cast<const char*>(&host_particlesIon[i].x), sizeof(float));
        ofsXIon.write(reinterpret_cast<const char*>(&host_particlesIon[i].y), sizeof(float));
        ofsXIon.write(reinterpret_cast<const char*>(&host_particlesIon[i].z), sizeof(float));
    }

    std::ofstream ofsXElectron(filenameXElectron, std::ios::binary);
    ofsXElectron << std::fixed << std::setprecision(6);
    for (int i = 0; i < totalNumElectron; i++) {
        ofsXElectron.write(reinterpret_cast<const char*>(&host_particlesElectron[i].x), sizeof(float));
        ofsXElectron.write(reinterpret_cast<const char*>(&host_particlesElectron[i].y), sizeof(float));
        ofsXElectron.write(reinterpret_cast<const char*>(&host_particlesElectron[i].z), sizeof(float));
    }


    float vx, vy, vz, KineticEnergy = 0.0f;

    std::ofstream ofsVIon(filenameVIon, std::ios::binary);
    ofsVIon << std::fixed << std::setprecision(6);
    for (int i = 0; i < totalNumIon; i++) {
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
    for (int i = 0; i < totalNumElectron; i++) {
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

