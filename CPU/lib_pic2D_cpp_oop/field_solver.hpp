#include <vector>
#include "const.hpp"

class FieldSolver
{
private:

public:
    void timeEvolutionB(
        std::vector<std::vector<std::vector<double>>>& B, 
        const std::vector<std::vector<std::vector<double>>>& E, 
        double dt
    );

    void timeEvolutionE(
        std::vector<std::vector<std::vector<double>>>& E, 
        const std::vector<std::vector<std::vector<double>>>& B, 
        const std::vector<std::vector<std::vector<double>>>& current, 
        double dt
    );

private:

};


