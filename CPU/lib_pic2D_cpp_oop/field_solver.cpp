#include "field_solver.hpp"


void FieldSolver::timeEvolutionB(
    std::vector<std::vector<std::vector<double>>>& B, 
    const std::vector<std::vector<std::vector<double>>>& E, 
    double dt
)
{
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            B[0][i][j] += -(E[2][i][(j + 1) % ny] - E[2][i][j]) / dy * dt;
            B[1][i][j] += (E[2][(i + 1) % nx][j] - E[2][i][j]) / dx * dt;
            B[2][i][j] += (-(E[1][(i + 1) % nx][j] - E[1][i][j]) / dx
                        + (E[0][i][(j + 1) % ny] - E[0][i][j]) / dy) * dt;
        }
    }
}


void FieldSolver::timeEvolutionE(
    std::vector<std::vector<std::vector<double>>>& E,
    const std::vector<std::vector<std::vector<double>>>& B, 
    const std::vector<std::vector<std::vector<double>>>& current,  
    double dt
)
{
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            E[0][i][j] += (-current[0][i][j] / epsilon0
                        + c * c * (B[2][i][j] - B[2][i][(j - 1 + ny) % ny]) / dy) * dt;
            E[1][i][j] += (-current[1][i][j] / epsilon0 
                        - c * c * (B[2][i][j] - B[2][(i - 1 + nx) % nx][j]) / dx) * dt;
            E[2][i][j] += (-current[2][i][j] / epsilon0 
                        + c * c * ((B[1][i][j] - B[1][(i - 1 + nx) % nx][j]) / dx
                        - (B[0][i][j] - B[0][i][(j - 1 + ny) % ny]) / dy)) * dt;
        }
    }
}


