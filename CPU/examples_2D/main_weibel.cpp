#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "../../lib_pic2D_cpp_oop/pic2D.hpp"


const double c = 1.0;
const double epsilon0 = 1.0;
const double mu0 = 1.0;

const int nx = 256;
const double dx = 1.0;
const double xmin = 0.0; 
const double xmax = nx * dx;

const int ny = 256;
const double dy = 1.0;
const double ymin = 0.0; 
const double ymax = ny * dy;

const double dt = 0.5;

const int numberDensityIon = 20;
const int numberDensityElectron = 20;

const int totalNumIon = nx * ny * numberDensityIon;
const int totalNumElectron = nx * ny * numberDensityElectron;
const int totalNumParticles = totalNumIon + totalNumElectron;

const double B0 = 1.0;

const double mRatio = 1.0;
const double mElectron = 1.0;
const double mIon = mRatio * mElectron;

const double tRatio = 1.0;
const double tElectron = 0.5 * mElectron * pow(0.1 * c, 2);
const double tIon = 0.5 * mIon * pow(0.1 * c, 2);

const double qRatio = -1.0;
const double qElectron = -1.0 * sqrt(epsilon0 * tElectron / static_cast<double>(numberDensityElectron));
const double qIon = qRatio * qElectron;

const double omegaPe = sqrt(static_cast<double>(numberDensityElectron) * pow(qElectron, 2) / mElectron / epsilon0);
const double omegaPi = sqrt(static_cast<double>(numberDensityIon) * pow(qIon, 2) / mIon / epsilon0);
const double omegaCe = abs(qElectron * B0 / mElectron);
const double omegaCi = qIon * B0 / mIon;

const double debyeLength = sqrt(epsilon0 * tElectron / static_cast<double>(numberDensityElectron) / pow(qElectron, 2));

const double vThIon = sqrt(tIon / mIon);
const double vThElectron = sqrt(tElectron / mElectron);
const double bulkVxIon = 0.0;
const double bulkVyIon = 0.0;
const double bulkVzIon = 0.0;
const double bulkVxElectron = 0.0;
const double bulkVyElectron = 0.0;
const double bulkVzElectron = 0.0;

const int totalStep = 3000;
double totalTime = 0.0;


void PIC2D::initialize()
{
    initializeParticle.uniformForPositionX(
        0, totalNumIon, 0, particlesIon
    );
    initializeParticle.uniformForPositionX(
        0, totalNumElectron, 100, particlesElectron
    );
    initializeParticle.uniformForPositionY(
        0, totalNumIon, 200, particlesIon
    );
    initializeParticle.uniformForPositionY(
        0, totalNumElectron, 300, particlesElectron
    );
    for (int i = 0; i < totalNumIon; i++) {
        particlesIon[i].z = 0.0;
    }
    for (int i = 0; i < totalNumElectron; i++) {
        particlesElectron[i].z = 0.0;
    }

    initializeParticle.maxwellDistributionForVelocity(
        bulkVxIon, bulkVyIon, bulkVzIon, vThIon, vThIon, 5.0 * vThIon, 
        0, totalNumIon, 400, particlesIon
    );
    initializeParticle.maxwellDistributionForVelocity(
        bulkVxElectron, bulkVyElectron, bulkVzElectron, vThElectron, vThElectron, 5.0 * vThElectron, 
        0, totalNumElectron, 500, particlesElectron
    );

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            B[0][i][j] = 0.0;
            B[1][i][j] = 0.0;
            B[2][i][j] = 0.0;
            E[0][i][j] = 0.0;
            E[1][i][j] = 0.0;
            E[2][i][j] = 0.0;
            current[0][i][j] = 0.0;
            current[1][i][j] = 0.0;
            current[2][i][j] = 0.0;
        }
    }
}


int main()
{
    std::string directoryname = "results";
    std::string filenameWithoutStep = "weibel";
    std::ofstream logfile("log.txt");
    int recordStep = 100;

    std::cout << "total number of partices is " << totalNumParticles << std::endl;
    std::cout << std::setprecision(4) 
              << "omega_pe * t = " << totalStep * dt * omegaPe << std::endl;

    PIC2D pIC2D;

    pIC2D.initialize();

    for (int step = 0; step < totalStep+1; step++) {
        if (step % recordStep == 0) {
            std::cout << std::to_string(step) << " step done : total time is "
                      << std::setprecision(4) << totalTime
                      << std::endl;
            logfile << std::setprecision(6) << totalTime << std::endl;
            pIC2D.saveFields(
                directoryname, filenameWithoutStep, step
            );
            pIC2D.saveEnergy(
                directoryname, filenameWithoutStep, step
            );
            //pIC2D.saveParticle(
            //    directoryname, filenameWithoutStep, step
            //);
        }

        pIC2D.oneStep();

        totalTime += dt;
    }

    return 0;
}



