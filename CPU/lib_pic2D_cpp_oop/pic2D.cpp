#include <fstream>
#include <iomanip>
#include "pic2D.hpp"


void PIC2D::oneStep()
{
    fieldSolver.timeEvolutionB(B, E, dt/2.0);
    boundary.periodicBoundaryBX();
    boundary.periodicBoundaryBY();

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            tmpB[0][i][j] = 0.5 * (B[0][i][j] + B[0][i][(j - 1 + ny) % ny]);
            tmpB[1][i][j] = 0.5 * (B[1][i][j] + B[1][(i - 1 + nx) % nx][j]);
            tmpB[2][i][j] = 0.25 * (B[2][i][j] + B[2][(i - 1 + nx) % nx][j]
                          + B[2][i][(j - 1 + ny) % ny] + B[2][(i - 1 + nx) % nx][(j - 1 + ny) % ny]);
            tmpE[0][i][j] = 0.5 * (E[0][i][j] + E[0][(i - 1 + nx) % nx][j]);
            tmpE[1][i][j] = 0.5 * (E[1][i][j] + E[1][i][(j - 1 + ny) % ny]);
            tmpE[2][i][j] = E[2][i][j];
        }
    }

    particlePush.pushVelocityForPeriodicBoundary(
        particlesIon, particlesElectron, tmpB, tmpE, dt
    );

    particlePush.pushPosition(
        particlesIon, particlesElectron, dt/2.0
    );
    boundary.periodicBoundaryParticleX(
        particlesIon, particlesElectron
    );
    boundary.periodicBoundaryParticleY(
        particlesIon, particlesElectron
    );

    currentCalculater.resetCurrent(tmpCurrent);
    currentCalculater.calculateCurrentForPeriodicBoundary(
        tmpCurrent, particlesIon, particlesElectron
    );
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            current[0][i][j] = 0.5 * (tmpCurrent[0][i][j] + tmpCurrent[0][(i + 1) % nx][j]);
            current[1][i][j] = 0.5 * (tmpCurrent[1][i][j] + tmpCurrent[1][i][(j + 1) % ny]);
            current[2][i][j] = tmpCurrent[2][i][j];
        }
    }

    fieldSolver.timeEvolutionB(B, E, dt/2.0);
    boundary.periodicBoundaryBX();
    boundary.periodicBoundaryBY();

    fieldSolver.timeEvolutionE(E, B, current, dt);
    boundary.periodicBoundaryEX();
    boundary.periodicBoundaryEY();

    particlePush.pushPosition(
        particlesIon, particlesElectron, dt/2.0
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
    std::string filenameB, filenameE, filenameCurrent;

    filenameB = directoryname + "/"
             + filenameWithoutStep + "_B_" + std::to_string(step)
             + ".txt";
    filenameE = directoryname + "/"
             + filenameWithoutStep + "_E_" + std::to_string(step)
             + ".txt";
    filenameCurrent = directoryname + "/"
             + filenameWithoutStep + "_current_" + std::to_string(step)
             + ".txt";


    std::ofstream ofsB(filenameB);
    ofsB << std::setprecision(6);
    for (int comp = 0; comp < 3; comp++) {
        for (int i = 0; i < nx-1; i++) {
            for (int j = 0; j < ny; j++) {
                ofsB << B[comp][i][j] << ",";
            }
        }
        for (int j = 0; j < ny-1; j++) {
            ofsB << B[comp][nx - 1][j] << ",";
        }
        ofsB << B[comp][nx - 1][ny - 1];
        ofsB << std::endl;
    }

    std::ofstream ofsE(filenameE);
    ofsE << std::setprecision(6);
    for (int comp = 0; comp < 3; comp++) {
        for (int i = 0; i < nx-1; i++) {
            for (int j = 0; j < ny; j++) {
                ofsE << E[comp][i][j] << ",";
            }
        }
        for (int j = 0; j < ny-1; j++) {
            ofsE << E[comp][nx - 1][j] << ",";
        }
        ofsE << E[comp][nx - 1][ny - 1];
        ofsE << std::endl;
    }

    std::ofstream ofsCurrent(filenameCurrent);
    ofsCurrent << std::setprecision(6);
    for (int comp = 0; comp < 3; comp++) {
        for (int i = 0; i < nx-1; i++) {
            for (int j = 0; j < ny; j++) {
                ofsCurrent << current[comp][i][j] << ",";
            }
        }
        for (int j = 0; j < ny-1; j++) {
            ofsCurrent << current[comp][nx - 1][j] << ",";
        }
        ofsCurrent << current[comp][nx - 1][ny - 1];
        ofsCurrent << std::endl;
    }
}


void PIC2D::saveParticle(
    std::string directoryname, 
    std::string filenameWithoutStep, 
    int step
)
{
    std::string filenameXIon, filenameXElectron;
    std::string filenameVIon, filenameVElectron;
    double vx, vy, vz;

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
    }
}


void PIC2D::saveEnergy(
    std::string directoryname, 
    std::string filenameWithoutStep, 
    int step
)
{
    std::string filenameBEnergy, filenameEEnergy;
     std::string filenameKineticEnergy;
    double BEnergy = 0.0, EEnergy = 0.0;
    double vx, vy, vz, KineticEnergy = 0.0;

    filenameBEnergy = directoryname + "/"
             + filenameWithoutStep + "_BEnergy_" + std::to_string(step)
             + ".txt";
    filenameEEnergy = directoryname + "/"
             + filenameWithoutStep + "_EEnergy_" + std::to_string(step)
             + ".txt";
    filenameKineticEnergy = directoryname + "/"
             + filenameWithoutStep + "_KE_" + std::to_string(step)
             + ".txt";
    
    for (int comp = 0; comp < 3; comp++) {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                BEnergy += B[comp][i][j] * B[comp][i][j];
            }
        }
    }

    BEnergy += 0.5 / mu0;

    std::ofstream ofsBEnergy(filenameBEnergy);
    ofsBEnergy << std::setprecision(6);
    ofsBEnergy << BEnergy << std::endl;


    for (int comp = 0; comp < 3; comp++) {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                EEnergy += E[comp][i][j] * E[comp][i][j];
            }
        }
    }

    EEnergy *= 0.5 * epsilon0;

    std::ofstream ofsEEnergy(filenameEEnergy);
    ofsEEnergy << std::setprecision(6);
    ofsEEnergy << EEnergy << std::endl;


    for (int i = 0; i < totalNumIon; i++) {
        vx = particlesIon[i].vx;
        vy = particlesIon[i].vy;
        vz = particlesIon[i].vz;
        KineticEnergy += 0.5 * mIon * (vx * vx + vy * vy + vz * vz);
    }
    for (int i = 0; i < totalNumElectron; i++) {
        vx = particlesElectron[i].vx;
        vy = particlesElectron[i].vy;
        vz = particlesElectron[i].vz;
        KineticEnergy += 0.5 * mElectron * (vx * vx + vy * vy + vz * vz);
    }

    std::ofstream ofsKineticEnergy(filenameKineticEnergy);
    ofsKineticEnergy << std::setprecision(6);
    ofsKineticEnergy << KineticEnergy << std::endl;
}

