#include "boundary.hpp"
#include <iostream>


// particle 

void Boundary::periodicBoundaryParticleX(
    std::vector<Particle>& particlesIon,
    std::vector<Particle>& particlesElectron
)
{
    for (int i = 0; i < totalNumIon; i++) {
        if (particlesIon[i].x < xmin) {
            particlesIon[i].x += xmax - xmin;
        }
        if (particlesIon[i].x > xmax) {
            particlesIon[i].x -= xmax - xmin;
        }
    }

    for (int i = 0; i < totalNumElectron; i++) {
        if (particlesElectron[i].x < xmin) {
            particlesElectron[i].x += xmax - xmin;
        }
        if (particlesElectron[i].x > xmax) {
            particlesElectron[i].x -= xmax - xmin;
        }
    }
}


void Boundary::periodicBoundaryParticleY(
    std::vector<Particle>& particlesIon,
    std::vector<Particle>& particlesElectron
)
{
    for (int i = 0; i < totalNumIon; i++) {
        if (particlesIon[i].y < ymin) {
            particlesIon[i].y += ymax - ymin;
        }
        if (particlesIon[i].y > ymax) {
            particlesIon[i].y -= ymax - ymin;
        }
    }

    for (int i = 0; i < totalNumElectron; i++) {
        if (particlesElectron[i].y < ymin) {
            particlesElectron[i].y += ymax - ymin;
        }
        if (particlesElectron[i].y > ymax) {
            particlesElectron[i].y -= ymax - ymin;
        }
    }
}


void Boundary::conductingWallBoundaryParticleX(
    std::vector<Particle>& particlesIon,
    std::vector<Particle>& particlesElectron
)
{
    for (int i = 0; i < totalNumIon; i++) {
        if (particlesIon[i].x < xmin) {
            particlesIon[i].x = 2.0 * xmin - particlesIon[i].x;
            particlesIon[i].vx = -1.0 * particlesIon[i].vx;
        }
        if (particlesIon[i].x > xmax) {
            particlesIon[i].x = 2.0 * xmax - particlesIon[i].x;
            particlesIon[i].vx = -1.0 * particlesIon[i].vx;
        }
    }

    for (int i = 0; i < totalNumElectron; i++) {
        if (particlesElectron[i].x < xmin) {
            particlesElectron[i].x = 2.0 * xmin - particlesElectron[i].x;
            particlesElectron[i].vx = -1.0 * particlesElectron[i].vx;
        }
        if (particlesElectron[i].x > xmax) {
            particlesElectron[i].x = 2.0 * xmax - particlesElectron[i].x;
            particlesElectron[i].vx = -1.0 * particlesElectron[i].vx;
        }
    }
}


void Boundary::conductingWallBoundaryParticleY(
    std::vector<Particle>& particlesIon,
    std::vector<Particle>& particlesElectron
)
{
    for (int i = 0; i < totalNumIon; i++) {
        if (particlesIon[i].y < ymin) {
            particlesIon[i].y = 2.0 * ymin - particlesIon[i].y;
            particlesIon[i].vy = -1.0 * particlesIon[i].vy;
        }
        if (particlesIon[i].y > ymax) {
            particlesIon[i].y = 2.0 * ymax - particlesIon[i].y;
            particlesIon[i].vy = -1.0 * particlesIon[i].vy;
        }
    }

    for (int i = 0; i < totalNumElectron; i++) {
        if (particlesElectron[i].y < ymin) {
            particlesElectron[i].y = 2.0 * ymin - particlesElectron[i].y;
            particlesElectron[i].vy = -1.0 * particlesElectron[i].vy;
        }
        if (particlesElectron[i].y > ymax) {
            particlesElectron[i].y = 2.0 * ymax - particlesElectron[i].y;
            particlesElectron[i].vy = -1.0 * particlesElectron[i].vy;
        }
    }
}

// B

void Boundary::periodicBoundaryBX()
{
    return;
}


void Boundary::periodicBoundaryBY()
{
    return;
}


void Boundary::conductingWallBoundaryBX(
    std::vector<std::vector<std::vector<double>>>& B
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}


void Boundary::conductingWallBoundaryBY(
    std::vector<std::vector<std::vector<double>>>& B
)
{
    for (int i = 0; i < nx; i++) {
        B[1][i][0] = -1.0 * B[1][i][1];
        B[1][i][ny - 1] = 0.0;
        B[2][i][0] = 0.5 * (B[2][i][0] + B[2][i][1]);
        B[2][i][ny - 1] = B[2][i][ny - 2];
    }
    for (int i = 0; i < nx-1; i++) {
        B[0][i+1][0] = -(B[1][i][1] - B[1][i][0]) / dy * dx + B[0][i][0];
        B[0][i+1][ny-2] = -(B[1][i][ny-1] - B[1][i][ny-2]) / dy * dx + B[0][i][ny-2];
        B[0][i+1][ny-1] = -(0.0 - B[1][i][ny-1]) / dy * dx + B[0][i][ny-1];
    }
}


void Boundary::symmetricWallBoundaryBX(
    std::vector<std::vector<std::vector<double>>>& B
)
{
    for (int j = 0; j < ny; j++) {
        B[0][0][j] = B[0][1][j];
        B[0][nx - 1][j] = 0.5 * (B[0][nx - 1][j] + B[0][nx - 2][j]);
        B[1][0][j] = 0.0;
        B[1][nx - 1][j] = -1.0 * B[1][nx - 2][j];
        B[2][0][j] = 0.0;
        B[2][nx - 1][j] = -1.0 * B[2][nx - 2][j];
    }
}


void Boundary::symmetricWallBoundaryBY(
    std::vector<std::vector<std::vector<double>>>& B
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}


// E

void Boundary::periodicBoundaryEX()
{
    return;
}


void Boundary::periodicBoundaryEY()
{
    return;
}


void Boundary::conductingWallBoundaryEX(
    std::vector<std::vector<std::vector<double>>>& E
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}


void Boundary::conductingWallBoundaryEY(
    std::vector<std::vector<std::vector<double>>>& E
)
{
    for (int i = 0; i < nx; i++) {
        E[0][i][0] = -1.0 * E[0][i][1];
        E[0][i][ny - 1] = 0.0;
        E[1][i][0] = 0.0;
        E[1][i][ny - 1] = -1.0 * E[1][i][ny - 2];
        E[2][i][0] = -1.0 * E[2][i][1];
        E[2][i][ny - 1] = 0.0;
    }
}


void Boundary::symmetricWallBoundaryEX(
    std::vector<std::vector<std::vector<double>>>& E
)
{
    for (int j = 0; j < ny; j++) {
        E[0][0][j] = 0.0;
        E[0][nx - 1][j] = -1.0 * E[0][nx - 2][j];
        E[1][0][j] = E[1][1][j];
        E[1][nx - 1][j] = 0.5 * (E[1][nx - 1][j] + E[1][nx - 2][j]);
        E[2][0][j] = E[2][1][j];
        E[2][nx - 1][j] = 0.5 * (E[2][nx - 1][j] + E[2][nx - 2][j]);
    }
}


void Boundary::symmetricWallBoundaryEY(
    std::vector<std::vector<std::vector<double>>>& E
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}


// current

void Boundary::periodicBoundaryCurrentX()
{
    return;
}


void Boundary::periodicBoundaryCurrentY()
{
    return;
}


void Boundary::conductingWallBoundaryCurrentX(
    std::vector<std::vector<std::vector<double>>>& current
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}


void Boundary::conductingWallBoundaryCurrentY(
    std::vector<std::vector<std::vector<double>>>& current
)
{
    for (int comp = 0; comp < 3; comp++) {
        for (int i = 0; i < nx; i++) {
            current[comp][i][0] = current[comp][i][1];
            current[comp][i][ny - 1] = current[comp][i][ny - 2];
        }
    }
}


void Boundary::symmetricWallBoundaryCurrentX(
    std::vector<std::vector<std::vector<double>>>& current
)
{
    for (int comp = 0; comp < 3; comp++) {
        for (int j = 0; j < ny; j++) {
            current[comp][0][j] = current[comp][1][j];
            current[comp][nx - 1][j] = current[comp][nx - 2][j];
        }
    }
}


void Boundary::symmetricWallBoundaryCurrentY(
    std::vector<std::vector<std::vector<double>>>& current
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}
