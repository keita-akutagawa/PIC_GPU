#include <cmath>
#include "filter.hpp"


void Filter::resetRho()
{
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            rho[i][j] = 0.0;
        }
    }
}


void Filter::calculateRhoOfOneSpecies(
    const std::vector<Particle>& particlesSpecies, 
    double q, int nStart, int nEnd
)
{
    double cx1, cx2; 
    int xIndex1, xIndex2;
    double cy1, cy2; 
    int yIndex1, yIndex2;
    double xOverDx, yOverDy;
    int wallXCondition, wallYCondition;

    for (int i = nStart; i < nEnd; i++) {
        xOverDx = particlesSpecies[i].x / dx;
        yOverDy = particlesSpecies[i].y / dy;

        xIndex1 = std::floor(xOverDx);
        xIndex2 = xIndex1 + 1;
        xIndex2 = (xIndex2 == nx) ? 0 : xIndex2;
        wallXCondition = (xIndex2 == 0) ? 0 : 1;
        yIndex1 = std::floor(yOverDy);
        yIndex2 = yIndex1 + 1;
        yIndex2 = (yIndex2 == ny) ? 0 : yIndex2;
        wallYCondition = (yIndex2 == 0) ? 0 : 1;

        cx1 = xOverDx - xIndex1;
        cx2 = 1.0 - cx1;
        cy1 = yOverDy - yIndex1;
        cy2 = 1.0 - cy1;

        rho[xIndex1][yIndex1] += q * cx2 * cy2;
        rho[xIndex2][yIndex1] += q * cx1 * cy2 * wallXCondition;
        rho[xIndex1][yIndex2] += q * cx2 * cy1 * wallYCondition;
        rho[xIndex2][yIndex2] += q * cx1 * cy1 * wallXCondition * wallYCondition;
    }
}


void Filter::calculateRho(
    const std::vector<Particle>& particlesIon, 
    const std::vector<Particle>& particlesElectron
)
{
    calculateRhoOfOneSpecies(particlesIon, qIon, 0, totalNumIon);
    calculateRhoOfOneSpecies(particlesElectron, qElectron, 0, totalNumElectron);
}



void Filter::langdonMarderCorrection(
    std::vector<std::vector<double>>& F, 
    std::vector<std::vector<std::vector<double>>>& E, 
    const std::vector<Particle>& particlesIon, 
    const std::vector<Particle>& particlesElectron
)
{
    resetRho();
    calculateRho(particlesIon, particlesElectron);

    for (int i = 1; i < nx; i++) {
        for (int j = 1; j < ny; j++) {
            F[i][j] = ((E[0][i][j] - E[0][i - 1][j])/dx + (E[1][i][j] - E[1][i][j - 1])/dy)
                    - rho[i][j] / epsilon0;
        }
    }
    for (int j = 0; j < ny; j++) {
        F[0][j] = ((E[0][0][j] - 0.0)/dx + (E[1][0][j] - 0.0)/dy)
                - rho[0][j] / epsilon0;
    }
    for (int i = 0; i < nx; i++) {
        F[i][0] = 0.0;
        F[i][ny - 1] = 0.0;
    }

    for (int i = 0; i < nx-1; i++) {
        for (int j = 0; j < ny-1; j++) {
            E[0][i][j] += dt * dOfLangdonMarderCorrection * (F[i + 1][j] - F[i][j]) / dx;
            E[1][i][j] += dt * dOfLangdonMarderCorrection * (F[i][j + 1] - F[i][j]) / dy;
        }
    }
    for (int j = 0; j < ny-1; j++) {
        E[0][nx - 1][j] += dt * dOfLangdonMarderCorrection * (0.0 - F[nx - 1][j]) / dx;
        E[1][nx - 1][j] += dt * dOfLangdonMarderCorrection * (F[nx - 1][j + 1] - F[nx - 1][j]) / dy;
    }
    for (int i = 0; i < nx-1; i++) {
        E[0][i][ny - 1] += dt * dOfLangdonMarderCorrection * (F[i + 1][ny - 1] - F[i][ny - 1]) / dx;
        E[1][i][ny - 1] += dt * dOfLangdonMarderCorrection * (0.0 - F[i][ny - 1]) / dy;
    }
    E[0][nx - 1][ny - 1] += dt * dOfLangdonMarderCorrection * (0.0 - F[nx - 1][ny - 1]) / dx;
    E[1][nx - 1][ny - 1] += dt * dOfLangdonMarderCorrection * (0.0 - F[nx - 1][ny - 1]) / dy;
}

