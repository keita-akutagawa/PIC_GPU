#include <cmath>
#include "current_calculater.hpp"
#include <iostream>


void CurrentCalculater::resetCurrent(
    std::vector<std::vector<std::vector<double>>>& current
)
{
    for (int comp = 0; comp < 3; comp++) {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                current[comp][i][j] = 0.0;
            }
        }
    }
}


void CurrentCalculater::calculateCurrentForPeriodicBoundary(
    std::vector<std::vector<std::vector<double>>>& current, 
    const std::vector<Particle>& particlesIon, 
    const std::vector<Particle>& particlesEleectron
)
{
    calculateCurrentOfOneSpeciesForPeriodicBoundary(
        current, particlesIon, qIon, totalNumIon
    );
    calculateCurrentOfOneSpeciesForPeriodicBoundary(
        current, particlesEleectron, qElectron, totalNumElectron
    );
}


void CurrentCalculater::calculateCurrentOfOneSpeciesForPeriodicBoundary(
    std::vector<std::vector<std::vector<double>>>& current, 
    const std::vector<Particle>& particlesSpecies,  
    double q, int totalNumSpecies
)
{
    double cx1, cx2; 
    int xIndex1, xIndex2;
    double cy1, cy2; 
    int yIndex1, yIndex2;
    double xOverDx, yOverDy;
    double qOverGamma, qVxOverGamma, qVyOverGamma, qVzOverGamma;

    for (int i = 0; i < totalNumSpecies; i++) {
        xOverDx = particlesSpecies[i].x / dx;
        yOverDy = particlesSpecies[i].y / dy;

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

        qOverGamma = q / particlesSpecies[i].gamma;
        qVxOverGamma = qOverGamma * particlesSpecies[i].vx;
        qVyOverGamma = qOverGamma * particlesSpecies[i].vy;
        qVzOverGamma = qOverGamma * particlesSpecies[i].vz;

        current[0][xIndex1][yIndex1] += qVxOverGamma * cx2 * cy2;
        current[0][xIndex2][yIndex1] += qVxOverGamma * cx1 * cy2;
        current[0][xIndex1][yIndex2] += qVxOverGamma * cx2 * cy1;
        current[0][xIndex2][yIndex2] += qVxOverGamma * cx1 * cy1;

        current[1][xIndex1][yIndex1] += qVyOverGamma * cx2 * cy2;
        current[1][xIndex2][yIndex1] += qVyOverGamma * cx1 * cy2;
        current[1][xIndex1][yIndex2] += qVyOverGamma * cx2 * cy1;
        current[1][xIndex2][yIndex2] += qVyOverGamma * cx1 * cy1;

        current[2][xIndex1][yIndex1] += qVzOverGamma * cx2 * cy2;
        current[2][xIndex2][yIndex1] += qVzOverGamma * cx1 * cy2;
        current[2][xIndex1][yIndex2] += qVzOverGamma * cx2 * cy1;
        current[2][xIndex2][yIndex2] += qVzOverGamma * cx1 * cy1;
    }
}



// 反射境界(壁) ----------------------------------------------------

void CurrentCalculater::calculateCurrentForWallBoundary(
    std::vector<std::vector<std::vector<double>>>& current, 
    const std::vector<Particle>& particlesIon, 
    const std::vector<Particle>& particlesEleectron
)
{
    calculateCurrentOfOneSpeciesForWallBoundary(
        current, particlesIon, qIon, totalNumIon
    );
    calculateCurrentOfOneSpeciesForWallBoundary(
        current, particlesEleectron, qElectron, totalNumElectron
    );
}


void CurrentCalculater::calculateCurrentOfOneSpeciesForWallBoundary(
    std::vector<std::vector<std::vector<double>>>& current, 
    const std::vector<Particle>& particlesSpecies,  
    double q, int totalNumSpecies
)
{
    double cx1, cx2; 
    int xIndex1, xIndex2;
    double cy1, cy2; 
    int yIndex1, yIndex2;
    double xOverDx, yOverDy;
    int wallXCondition, wallYCondition;
    double qOverGamma, qVxOverGamma, qVyOverGamma, qVzOverGamma;

    for (int i = 0; i < totalNumSpecies; i++) {
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

        qOverGamma = q / particlesSpecies[i].gamma;
        qVxOverGamma = qOverGamma * particlesSpecies[i].vx;
        qVyOverGamma = qOverGamma * particlesSpecies[i].vy;
        qVzOverGamma = qOverGamma * particlesSpecies[i].vz;

        current[0][xIndex1][yIndex1] += qVxOverGamma * cx2 * cy2;
        current[0][xIndex2][yIndex1] += qVxOverGamma * cx1 * cy2 * wallXCondition;
        current[0][xIndex1][yIndex2] += qVxOverGamma * cx2 * cy1 * wallYCondition;
        current[0][xIndex2][yIndex2] += qVxOverGamma * cx1 * cy1 * wallXCondition * wallYCondition;

        current[1][xIndex1][yIndex1] += qVyOverGamma * cx2 * cy2;
        current[1][xIndex2][yIndex1] += qVyOverGamma * cx1 * cy2 * wallXCondition;
        current[1][xIndex1][yIndex2] += qVyOverGamma * cx2 * cy1 * wallYCondition;
        current[1][xIndex2][yIndex2] += qVyOverGamma * cx1 * cy1 * wallXCondition * wallYCondition;

        current[2][xIndex1][yIndex1] += qVzOverGamma * cx2 * cy2;
        current[2][xIndex2][yIndex1] += qVzOverGamma * cx1 * cy2 * wallXCondition;
        current[2][xIndex1][yIndex2] += qVzOverGamma * cx2 * cy1 * wallYCondition;
        current[2][xIndex2][yIndex2] += qVzOverGamma * cx1 * cy1 * wallXCondition * wallYCondition;
    }
}


