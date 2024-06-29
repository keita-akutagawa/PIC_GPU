#include "field_solver.hpp"


void FieldSolver::timeEvolutionB(
    std::vector<std::vector<double>>& B, 
    const std::vector<std::vector<double>>& E, 
    double dt
)
{
    for (int i = 0; i < nx-1; i++) {
        //B[0][i] += 0.0;
        B[1][i] += (E[2][i+1] - E[2][i]) / dx * dt;
        B[2][i] += -(E[1][i+1] - E[1][i]) / dx * dt;
    }
    //周期境界条件
    //B[0][nx-1] += 0.0;
    B[1][nx-1] += (E[2][0] - E[2][nx-1]) / dx * dt;
    B[2][nx-1] += -(E[1][0] - E[1][nx-1]) / dx * dt;
}


void FieldSolver::timeEvolutionE(
    std::vector<std::vector<double>>& E,
    const std::vector<std::vector<double>>& B, 
    const std::vector<std::vector<double>>& current,  
    double dt
)
{
    for (int i = 1; i < nx; i++) {
        E[0][i] += (-current[0][i] / epsilon0) * dt;
        E[1][i] += (-current[1][i] / epsilon0 
                    - c * c * (B[2][i] - B[2][i-1]) / dx) * dt;
        E[2][i] += (-current[2][i] / epsilon0 
                    + c * c * (B[1][i] - B[1][i-1]) / dx) * dt;
    }
    //周期境界条件
    E[0][0] += (-current[0][0] / epsilon0) * dt;
    E[1][0] += (-current[1][0] / epsilon0 
                - c * c * (B[2][0] - B[2][nx-1]) / dx) * dt;
    E[2][0] += (-current[2][0] / epsilon0 
                + c * c * (B[1][0] - B[1][nx-1]) / dx) * dt;
}


