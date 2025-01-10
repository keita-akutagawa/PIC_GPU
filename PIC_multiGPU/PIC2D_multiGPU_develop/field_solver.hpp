#include <thrust/device_vector.h>
#include "field_parameter_struct.hpp"
#include "const.hpp"
#include "mpi.hpp"


class FieldSolver
{
private:
    MPIInfo& mPIInfo; 

public:
    FieldSolver(MPIInfo& mPIInfo);

    void timeEvolutionB(
        thrust::device_vector<MagneticField>& B, 
        const thrust::device_vector<ElectricField>& E, 
        const float dt
    );

    void timeEvolutionE(
        thrust::device_vector<ElectricField>& E, 
        const thrust::device_vector<MagneticField>& B, 
        const thrust::device_vector<CurrentField>& current, 
        const float dt
    );

private:

};


