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
    std::string filenameB, filenameE, filenameCurrent;
    std::string filenameBEnergy, filenameEEnergy;
    double BEnergy = 0.0, EEnergy = 0.0;

    filenameB = directoryname + "/"
             + filenameWithoutStep + "_B_" + std::to_string(step)
             + ".txt";
    filenameE = directoryname + "/"
             + filenameWithoutStep + "_E_" + std::to_string(step)
             + ".txt";
    filenameCurrent = directoryname + "/"
             + filenameWithoutStep + "_current_" + std::to_string(step)
             + ".txt";
    filenameBEnergy = directoryname + "/"
             + filenameWithoutStep + "_BEnergy_" + std::to_string(step)
             + ".txt";
    filenameEEnergy = directoryname + "/"
             + filenameWithoutStep + "_EEnergy_" + std::to_string(step)
             + ".txt";


    std::ofstream ofsB(filenameB);
    ofsB << std::setprecision(6);
    for (int comp = 0; comp < 3; comp++) {
        for (int i = 0; i < nx-1; i++) {
            ofsB << B[comp][i] << ",";
            BEnergy += B[comp][i] * B[comp][i];
        }
        ofsB << B[comp][nx-1];
        ofsB << std::endl;
        BEnergy += B[comp][nx-1] * B[comp][nx-1];
    }
    BEnergy += 0.5 / mu0;

    std::ofstream ofsE(filenameE);
    ofsE << std::setprecision(6);
    for (int comp = 0; comp < 3; comp++) {
        for (int i = 0; i < nx-1; i++) {
            ofsE << E[comp][i] << ",";
            EEnergy += E[comp][i] * E[comp][i];
        }
        ofsE << E[comp][nx-1];
        ofsE << std::endl;
        EEnergy += E[comp][nx-1] * E[comp][nx-1];
    }
    EEnergy *= 0.5 * epsilon0;

    std::ofstream ofsCurrent(filenameCurrent);
    ofsCurrent << std::setprecision(6);
    for (int comp = 0; comp < 3; comp++) {
        for (int i = 0; i < nx-1; i++) {
            ofsCurrent << current[comp][i] << ",";
        }
        ofsCurrent << current[comp][nx-1];
        ofsCurrent << std::endl;
    }

    std::ofstream ofsBEnergy(filenameBEnergy);
    ofsBEnergy << std::setprecision(6);
    ofsBEnergy << BEnergy << std::endl;

    std::ofstream ofsEEnergy(filenameEEnergy);
    ofsEEnergy << std::setprecision(6);
    ofsEEnergy << EEnergy << std::endl;
}


void PIC1D::saveParticle(
    std::string directoryname, 
    std::string filenameWithoutStep, 
    int step
)
{
    std::string filenameXIon, filenameXElectron;
    std::string filenameVIon, filenameVElectron;
    std::string filenameKineticEnergy;
    double vx, vy, vz, KineticEnergy = 0.0;

    filenameXIon = directoryname + "/"
             + filenameWithoutStep + "_x_ion_" + std::to_string(step)
             + ".txt";
    filenameXElectron = directoryname + "/"
             + filenameWithoutStep + "_x_electron_" + std::to_string(step)
             + ".txt";
    filenameVIon = directoryname + "/"
             + filenameWithoutStep + "_v_ion_" + std::to_string(step)
             + ".txt";
    filenameVElectron = directoryname + "/"
             + filenameWithoutStep + "_v_electron_" + std::to_string(step)
             + ".txt";
    filenameKineticEnergy = directoryname + "/"
             + filenameWithoutStep + "_KE_" + std::to_string(step)
             + ".txt";


    std::ofstream ofsXIon(filenameXIon);
    ofsXIon << std::setprecision(6);
    for (int i = 0; i < totalNumIon; i++) {
        ofsXIon << particlesIon[i].x << "," 
                << particlesIon[i].y << "," 
                << particlesIon[i].z << std::endl ;
    }

    std::ofstream ofsXElectron(filenameXElectron);
    ofsXElectron << std::setprecision(6);
    for (int i = 0; i < totalNumElectron; i++) {
        ofsXElectron << particlesElectron[i].x << "," 
                     << particlesElectron[i].y << "," 
                     << particlesElectron[i].z << std::endl ;
    }

    std::ofstream ofsVIon(filenameVIon);
    ofsVIon << std::setprecision(6);
    for (int i = 0; i < totalNumIon; i++) {
        vx = particlesIon[i].vx;
        vy = particlesIon[i].vy;
        vz = particlesIon[i].vz;

        ofsVIon << vx << "," 
                << vy << "," 
                << vz << std::endl;

        KineticEnergy += 0.5 * mIon * (vx * vx + vy * vy + vz * vz);
    }

    std::ofstream ofsVElectron(filenameVElectron);
    ofsVElectron << std::setprecision(6);
    for (int i = 0; i < totalNumElectron; i++) {
        vx = particlesElectron[i].vx;
        vy = particlesElectron[i].vy;
        vz = particlesElectron[i].vz;

        ofsVElectron << vx << "," 
                     << vy << "," 
                     << vz << std::endl;
        
        KineticEnergy += 0.5 * mElectron * (vx * vx + vy * vy + vz * vz);
    }

    std::ofstream ofsKineticEnergy(filenameKineticEnergy);
    ofsKineticEnergy << std::setprecision(6);
    ofsKineticEnergy << KineticEnergy << std::endl;
}

