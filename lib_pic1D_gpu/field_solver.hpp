#include <thrust/device_vector.h>
#include "field_parameter_struct.hpp"
#include "const.hpp"


class FieldSolver
{
private:

public:
    void timeEvolutionB(
        thrust::device_vector<MagneticField>& B, 
        const thrust::device_vector<ElectricField>& E, 
        const double dt
    );

    void timeEvolutionE(
        thrust::device_vector<ElectricField>& E, 
        const thrust::device_vector<MagneticField>& B, 
        const thrust::device_vector<CurrentField>& current, 
        const double dt
    );

private:

};


