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
    MagneticField* tmpMagneticField, ElectricField* tmpElectricField, 
    const MagneticField* magneticField, const ElectricField* electricField
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (0 < i && i < device_nx) {
        tmpMagneticField[i].bX = magneticField[i].bX;
        tmpMagneticField[i].bY = 0.5 * (magneticField[i].bY + magneticField[i - 1].bY);
        tmpMagneticField[i].bZ = 0.5 * (magneticField[i].bZ + magneticField[i - 1].bZ);
        tmpElectricField[i].eX = 0.5 * (electricField[i].eX + electricField[i - 1].eX);
        tmpElectricField[i].eY = electricField[i].eY;
        tmpElectricField[i].eZ = electricField[i].eZ;
    }
}

__global__ void getHalfCurrent_kernel(
    CurrentField* currentField, const CurrentField* tmpCurrentField
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < device_nx - 1) {
        currentField[i].jX = 0.5 * (tmpCurrentField[i].jX + tmpCurrentField[i + 1].jX);
        currentField[i].jY = tmpCurrentField[i].jY;
        currentField[i].jZ = tmpCurrentField[i].jZ;
    }
}


void PIC1D::oneStep()
{
    fieldSolver.timeEvolutionB(B, E, dt/2.0);


    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x);

    getCenterBE_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpB.data()), 
        thrust::raw_pointer_cast(tmpE.data()), 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data())
    );

    cudaDeviceSynchronize();


    particlePush.pushVelocity(
        particlesIon, particlesElectron, tmpB, tmpE, dt
    );


    particlePush.pushPosition(
        particlesIon, particlesElectron, dt/2.0
    );
    boundary.periodicBoundaryParticleX(
        particlesIon, particlesElectron
    );


    currentCalculater.resetCurrent(tmpCurrent);
    currentCalculater.calculateCurrent(
        tmpCurrent, particlesIon, particlesElectron
    );
    getHalfCurrent_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(tmpCurrent.data())
    );


    fieldSolver.timeEvolutionB(B, E, dt/2.0);


    fieldSolver.timeEvolutionE(E, B, current, dt);


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
        ofsB.write(reinterpret_cast<const char*>(&host_B[i].bX), sizeof(float));
        ofsB.write(reinterpret_cast<const char*>(&host_B[i].bY), sizeof(float));
        ofsB.write(reinterpret_cast<const char*>(&host_B[i].bZ), sizeof(float));
        BEnergy += host_B[i].bX * host_B[i].bX + host_B[i].bY * host_B[i].bY + host_B[i].bZ * host_B[i].bZ;
    }
    BEnergy *= 0.5f / mu0;

    std::ofstream ofsE(filenameE, std::ios::binary);
    ofsE << std::fixed << std::setprecision(6);
    for (int i = 0; i < nx; i++) {
        ofsE.write(reinterpret_cast<const char*>(&host_E[i].eX), sizeof(float));
        ofsE.write(reinterpret_cast<const char*>(&host_E[i].eY), sizeof(float));
        ofsE.write(reinterpret_cast<const char*>(&host_E[i].eZ), sizeof(float));
        BEnergy += host_E[i].eX * host_E[i].eX + host_E[i].eY * host_E[i].eY + host_E[i].eZ * host_E[i].eZ;
    }
    EEnergy *= 0.5f * epsilon0;

    std::ofstream ofsCurrent(filenameCurrent, std::ios::binary);
    ofsCurrent << std::fixed << std::setprecision(6);
    for (int i = 0; i < nx; i++) {
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
        ofsXIon.write(reinterpret_cast<const char*>(&host_particleIon[i].x), sizeof(float));
        ofsXIon.write(reinterpret_cast<const char*>(&host_particleIon[i].y), sizeof(float));
        ofsXIon.write(reinterpret_cast<const char*>(&host_particleIon[i].z), sizeof(float));
    }

    std::ofstream ofsXElectron(filenameXElectron, std::ios::binary);
    ofsXElectron << std::fixed << std::setprecision(6);
    for (int i = 0; i < totalNumElectron; i++) {
        ofsXElectron.write(reinterpret_cast<const char*>(&host_particleElectron[i].x), sizeof(float));
        ofsXElectron.write(reinterpret_cast<const char*>(&host_particleElectron[i].y), sizeof(float));
        ofsXElectron.write(reinterpret_cast<const char*>(&host_particleElectron[i].z), sizeof(float));
    }


    float vx, vy, vz, KineticEnergy = 0.0f;

    std::ofstream ofsVIon(filenameVIon, std::ios::binary);
    ofsVIon << std::fixed << std::setprecision(6);
    for (int i = 0; i < totalNumIon; i++) {
        vx = host_particleIon[i].vx;
        vy = host_particleIon[i].vy;
        vz = host_particleIon[i].vz;

        ofsVIon.write(reinterpret_cast<const char*>(&vx), sizeof(float));
        ofsVIon.write(reinterpret_cast<const char*>(&vy), sizeof(float));
        ofsVIon.write(reinterpret_cast<const char*>(&vz), sizeof(float));

        KineticEnergy += 0.5f * mIon * (vx * vx + vy * vy + vz * vz);
    }

    std::ofstream ofsVElectron(filenameVElectron, std::ios::binary);
    ofsVElectron << std::fixed << std::setprecision(6);
    for (int i = 0; i < totalNumElectron; i++) {
        vx = host_particleElectron[i].vx;
        vy = host_particleElectron[i].vy;
        vz = host_particleElectron[i].vz;

        ofsVElectron.write(reinterpret_cast<const char*>(&vx), sizeof(float));
        ofsVElectron.write(reinterpret_cast<const char*>(&vy), sizeof(float));
        ofsVElectron.write(reinterpret_cast<const char*>(&vz), sizeof(float));
        
        KineticEnergy += 0.5f * mElectron * (vx * vx + vy * vy + vz * vz);
    }

    std::ofstream ofsKineticEnergy(filenameKineticEnergy, std::ios::binary);
    ofsKineticEnergy << std::fixed << std::setprecision(6);
    ofsKineticEnergy.write(reinterpret_cast<const char*>(&KineticEnergy), sizeof(float));
}

