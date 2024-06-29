#include <fstream>
#include <iomanip>
#include <cmath>
#include "pic2D_symX_conY.hpp"


void PIC2DSymXConY::oneStep()
{
    fieldSolver.timeEvolutionB(B, E, dt/2.0);
    boundary.symmetricWallBoundaryBX(B);
    boundary.conductingWallBoundaryBY(B);

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
    boundary.symmetricWallBoundaryBX(tmpB);
    boundary.conductingWallBoundaryBY(tmpB);
    boundary.symmetricWallBoundaryEX(tmpE);
    boundary.conductingWallBoundaryEY(tmpE);

    particlePush.pushVelocityForWallBoundary(
        particlesIon, particlesElectron, tmpB, tmpE, dt
    );

    particlePush.pushPosition(
        particlesIon, particlesElectron, dt/2.0
    );
    boundary.conductingWallBoundaryParticleX(
        particlesIon, particlesElectron
    );
    boundary.conductingWallBoundaryParticleY(
        particlesIon, particlesElectron
    );

    currentCalculater.resetCurrent(tmpCurrent);
    currentCalculater.calculateCurrentForWallBoundary(
        tmpCurrent, particlesIon, particlesElectron
    );
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            current[0][i][j] = 0.5 * (tmpCurrent[0][i][j] + tmpCurrent[0][(i + 1) % nx][j]);
            current[1][i][j] = 0.5 * (tmpCurrent[1][i][j] + tmpCurrent[1][i][(j + 1) % ny]);
            current[2][i][j] = tmpCurrent[2][i][j];
        }
    }
    boundary.symmetricWallBoundaryCurrentX(current);
    boundary.conductingWallBoundaryCurrentY(current);

    fieldSolver.timeEvolutionB(B, E, dt/2.0);
    boundary.symmetricWallBoundaryBX(B);
    boundary.conductingWallBoundaryBY(B);

    fieldSolver.timeEvolutionE(E, B, current, dt);
    boundary.symmetricWallBoundaryEX(E);
    boundary.conductingWallBoundaryEY(E);
    filter.langdonMarderCorrection(F, E, particlesIon, particlesElectron);
    boundary.symmetricWallBoundaryEX(E);
    boundary.conductingWallBoundaryEY(E);

    particlePush.pushPosition(
        particlesIon, particlesElectron, dt/2.0
    );
    boundary.conductingWallBoundaryParticleX(
        particlesIon, particlesElectron
    );
    boundary.conductingWallBoundaryParticleY(
        particlesIon, particlesElectron
    );
}


void PIC2DSymXConY::saveFields(
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


void PIC2DSymXConY::saveParticle(
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


void PIC2DSymXConY::saveEnergy(
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


void PIC2DSymXConY::saveMoments(
    std::string directoryname, 
    std::string filenameWithoutStep, 
    int step
)
{
    resetMoments();

    std::string filenameZerothMomentIon, filenameZerothMomentElectron;
    std::string filenameFirstMomentIon, filenameFirstMomentElectron;
    std::string filenameSecondMomentIon, filenameSecondMomentElectron;

    filenameZerothMomentIon = directoryname + "/"
                            + filenameWithoutStep + "_zeroth_moment_ion_" + std::to_string(step)
                            + ".txt";
    filenameZerothMomentElectron = directoryname + "/"
                                 + filenameWithoutStep + "_zeroth_moment_electron_" + std::to_string(step)
                                 + ".txt";
    filenameFirstMomentIon = directoryname + "/"
                            + filenameWithoutStep + "_first_moment_ion_" + std::to_string(step)
                            + ".txt";
    filenameFirstMomentElectron = directoryname + "/"
                                 + filenameWithoutStep + "_first_moment_electron_" + std::to_string(step)
                                 + ".txt";
    filenameSecondMomentIon = directoryname + "/"
                            + filenameWithoutStep + "_second_moment_ion_" + std::to_string(step)
                            + ".txt";
    filenameSecondMomentElectron = directoryname + "/"
                                 + filenameWithoutStep + "_second_moment_electron_" + std::to_string(step)
                                 + ".txt";
    

    calculateMomentOfOneSpecies(
        zerothMomentIon, firstMomentIon, secondMomentIon, 
        particlesIon, totalNumIon
    );
    calculateMomentOfOneSpecies(
        zerothMomentElectron, firstMomentElectron, secondMomentElectron, 
        particlesElectron, totalNumElectron
    );


    std::ofstream ofsZerothMomentIon(filenameZerothMomentIon);
    ofsZerothMomentIon << std::setprecision(6);
    for (int i = 0; i < nx-1; i++) {
        for (int j = 0; j < ny; j++) {
            ofsZerothMomentIon << zerothMomentIon[i][j] << ",";
        }
    }
    for (int j = 0; j < ny-1; j++) {
        ofsZerothMomentIon << zerothMomentIon[nx - 1][j] << ",";
    }
    ofsZerothMomentIon << zerothMomentIon[nx - 1][ny - 1];
    ofsZerothMomentIon << std::endl;

    std::ofstream ofsZerothMomentElectron(filenameZerothMomentElectron);
    ofsZerothMomentElectron << std::setprecision(6);
    for (int i = 0; i < nx-1; i++) {
        for (int j = 0; j < ny; j++) {
            ofsZerothMomentElectron << zerothMomentElectron[i][j] << ",";
        }
    }
    for (int j = 0; j < ny-1; j++) {
        ofsZerothMomentElectron << zerothMomentElectron[nx - 1][j] << ",";
    }
    ofsZerothMomentElectron << zerothMomentElectron[nx - 1][ny - 1];
    ofsZerothMomentElectron << std::endl;


    std::ofstream ofsFirstMomentIon(filenameFirstMomentIon);
    ofsFirstMomentIon << std::setprecision(6);
    for (int comp = 0; comp < 3; comp++) {
        for (int i = 0; i < nx-1; i++) {
            for (int j = 0; j < ny; j++) {
                ofsFirstMomentIon << firstMomentIon[comp][i][j] << ",";
            }
        }
        for (int j = 0; j < ny-1; j++) {
            ofsFirstMomentIon << firstMomentIon[comp][nx - 1][j] << ",";
        }
        ofsFirstMomentIon << firstMomentIon[comp][nx - 1][ny - 1];
        ofsFirstMomentIon << std::endl;
    }

    std::ofstream ofsFirstMomentElectron(filenameFirstMomentElectron);
    ofsFirstMomentElectron << std::setprecision(6);
    for (int comp = 0; comp < 3; comp++) {
        for (int i = 0; i < nx-1; i++) {
            for (int j = 0; j < ny; j++) {
                ofsFirstMomentElectron << firstMomentElectron[comp][i][j] << ",";
            }
        }
        for (int j = 0; j < ny-1; j++) {
            ofsFirstMomentElectron << firstMomentElectron[comp][nx - 1][j] << ",";
        }
        ofsFirstMomentElectron << firstMomentElectron[comp][nx - 1][ny - 1];
        ofsFirstMomentElectron << std::endl;
    }


    std::ofstream ofsSecondMomentIon(filenameSecondMomentIon);
    ofsSecondMomentIon << std::setprecision(6);
    for (int comp = 0; comp < 9; comp++) {
        for (int i = 0; i < nx-1; i++) {
            for (int j = 0; j < ny; j++) {
                ofsSecondMomentIon << secondMomentIon[comp][i][j] << ",";
            }
        }
        for (int j = 0; j < ny-1; j++) {
            ofsSecondMomentIon << secondMomentIon[comp][nx - 1][j] << ",";
        }
        ofsSecondMomentIon << secondMomentIon[comp][nx - 1][ny - 1];
        ofsSecondMomentIon << std::endl;
    }

    std::ofstream ofsSecondMomentElectron(filenameSecondMomentElectron);
    ofsSecondMomentElectron << std::setprecision(6);
    for (int comp = 0; comp < 9; comp++) {
        for (int i = 0; i < nx-1; i++) {
            for (int j = 0; j < ny; j++) {
                ofsSecondMomentElectron << secondMomentElectron[comp][i][j] << ",";
            }
        }
        for (int j = 0; j < ny-1; j++) {
            ofsSecondMomentElectron << secondMomentElectron[comp][nx - 1][j] << ",";
        }
        ofsSecondMomentElectron << secondMomentElectron[comp][nx - 1][ny - 1];
        ofsSecondMomentElectron << std::endl;
    }
}



void PIC2DSymXConY::resetMoments()
{
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            zerothMomentIon[i][j] = 0.0;
            zerothMomentElectron[i][j] = 0.0;
        }
    }

    for (int comp = 0; comp < 3; comp++) {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                firstMomentIon[comp][i][j] = 0.0;
                firstMomentElectron[comp][i][j] = 0.0;
            }
        }
    }

    for (int comp = 0; comp < 9; comp++) {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                secondMomentIon[comp][i][j] = 0.0;
                secondMomentElectron[comp][i][j] = 0.0;
            }
        }
    }
}


void PIC2DSymXConY::calculateMomentOfOneSpecies(
    std::vector<std::vector<double>>& zerothMomentSpecies, 
    std::vector<std::vector<std::vector<double>>>& firstMomentSpecies, 
    std::vector<std::vector<std::vector<double>>>& secondMomentSpecies, 
    const std::vector<Particle>& particlesSpecies, 
    int totalNumSpecies
)
{
    double x, y, vx, vy, vz;
    double cx1, cx2; 
    int xIndex1, xIndex2;
    double cy1, cy2; 
    int yIndex1, yIndex2;
    double xOverDx, yOverDy;

    for (int i = 0; i < totalNumSpecies; i++) {
        x = particlesSpecies[i].x;
        y = particlesSpecies[i].y;
        vx = particlesSpecies[i].vx;
        vy = particlesSpecies[i].vy;
        vz = particlesSpecies[i].vz;

        xOverDx = x / dx;
        yOverDy = y / dy;

        xIndex1 = std::floor(xOverDx);
        xIndex2 = xIndex1 + 1;
        xIndex2 = (xIndex2 == nx) ? 0 : xIndex2;
        yIndex1 = std::floor(yOverDy);
        yIndex2 = yIndex1 + 1;
        yIndex2 = (yIndex2 == ny) ? 0 : yIndex2;

        cx1 = xOverDx - xIndex1;
        cx2 = 1.0 - cx1;
        cy1 = yOverDy - yIndex1;
        cy2 = 1.0 - cy1;

        zerothMomentSpecies[xIndex1][yIndex1] += cx2 * cy2;
        zerothMomentSpecies[xIndex2][yIndex1] += cx1 * cy2;
        zerothMomentSpecies[xIndex1][yIndex2] += cx2 * cy1;
        zerothMomentSpecies[xIndex2][yIndex2] += cx1 * cy1;

        firstMomentSpecies[0][xIndex1][yIndex1] += vx * cx2 * cy2;
        firstMomentSpecies[0][xIndex2][yIndex1] += vx * cx1 * cy2;
        firstMomentSpecies[0][xIndex1][yIndex2] += vx * cx2 * cy1;
        firstMomentSpecies[0][xIndex2][yIndex2] += vx * cx1 * cy1;

        firstMomentSpecies[1][xIndex1][yIndex1] += vy * cx2 * cy2;
        firstMomentSpecies[1][xIndex2][yIndex1] += vy * cx1 * cy2;
        firstMomentSpecies[1][xIndex1][yIndex2] += vy * cx2 * cy1;
        firstMomentSpecies[1][xIndex2][yIndex2] += vy * cx1 * cy1;

        firstMomentSpecies[2][xIndex1][yIndex1] += vz * cx2 * cy2;
        firstMomentSpecies[2][xIndex2][yIndex1] += vz * cx1 * cy2;
        firstMomentSpecies[2][xIndex1][yIndex2] += vz * cx2 * cy1;
        firstMomentSpecies[2][xIndex2][yIndex2] += vz * cx1 * cy1;

        secondMomentSpecies[0][xIndex1][yIndex1] += vx * vx * cx2 * cy2;
        secondMomentSpecies[0][xIndex2][yIndex1] += vx * vx * cx1 * cy2;
        secondMomentSpecies[0][xIndex1][yIndex2] += vx * vx * cx2 * cy1;
        secondMomentSpecies[0][xIndex2][yIndex2] += vx * vx * cx1 * cy1;

        secondMomentSpecies[1][xIndex1][yIndex1] += vx * vy * cx2 * cy2;
        secondMomentSpecies[1][xIndex2][yIndex1] += vx * vy * cx1 * cy2;
        secondMomentSpecies[1][xIndex1][yIndex2] += vx * vy * cx2 * cy1;
        secondMomentSpecies[1][xIndex2][yIndex2] += vx * vy * cx1 * cy1;

        secondMomentSpecies[2][xIndex1][yIndex1] += vx * vz * cx2 * cy2;
        secondMomentSpecies[2][xIndex2][yIndex1] += vx * vz * cx1 * cy2;
        secondMomentSpecies[2][xIndex1][yIndex2] += vx * vz * cx2 * cy1;
        secondMomentSpecies[2][xIndex2][yIndex2] += vx * vz * cx1 * cy1;

        secondMomentSpecies[3][xIndex1][yIndex1] += vy * vx * cx2 * cy2;
        secondMomentSpecies[3][xIndex2][yIndex1] += vy * vx * cx1 * cy2;
        secondMomentSpecies[3][xIndex1][yIndex2] += vy * vx * cx2 * cy1;
        secondMomentSpecies[3][xIndex2][yIndex2] += vy * vx * cx1 * cy1;

        secondMomentSpecies[4][xIndex1][yIndex1] += vy * vy * cx2 * cy2;
        secondMomentSpecies[4][xIndex2][yIndex1] += vy * vy * cx1 * cy2;
        secondMomentSpecies[4][xIndex1][yIndex2] += vy * vy * cx2 * cy1;
        secondMomentSpecies[4][xIndex2][yIndex2] += vy * vy * cx1 * cy1;

        secondMomentSpecies[5][xIndex1][yIndex1] += vy * vz * cx2 * cy2;
        secondMomentSpecies[5][xIndex2][yIndex1] += vy * vz * cx1 * cy2;
        secondMomentSpecies[5][xIndex1][yIndex2] += vy * vz * cx2 * cy1;
        secondMomentSpecies[5][xIndex2][yIndex2] += vy * vz * cx1 * cy1;

        secondMomentSpecies[6][xIndex1][yIndex1] += vz * vx * cx2 * cy2;
        secondMomentSpecies[6][xIndex2][yIndex1] += vz * vx * cx1 * cy2;
        secondMomentSpecies[6][xIndex1][yIndex2] += vz * vx * cx2 * cy1;
        secondMomentSpecies[6][xIndex2][yIndex2] += vz * vx * cx1 * cy1;

        secondMomentSpecies[7][xIndex1][yIndex1] += vz * vy * cx2 * cy2;
        secondMomentSpecies[7][xIndex2][yIndex1] += vz * vy * cx1 * cy2;
        secondMomentSpecies[7][xIndex1][yIndex2] += vz * vy * cx2 * cy1;
        secondMomentSpecies[7][xIndex2][yIndex2] += vz * vy * cx1 * cy1;

        secondMomentSpecies[8][xIndex1][yIndex1] += vz * vz * cx2 * cy2;
        secondMomentSpecies[8][xIndex2][yIndex1] += vz * vz * cx1 * cy2;
        secondMomentSpecies[8][xIndex1][yIndex2] += vz * vz * cx2 * cy1;
        secondMomentSpecies[8][xIndex2][yIndex2] += vz * vz * cx1 * cy1;
    }
}

