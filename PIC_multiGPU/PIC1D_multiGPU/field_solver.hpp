#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "field_parameter_struct.hpp"
#include "const.hpp"
#include "mpi.hpp"


class FieldSolver
{
private:

public:
    void timeEvolutionB(
        thrust::device_vector<MagneticField>& B, 
        thrust::device_vector<ElectricField>& E, 
        const double dt, 
        MPIInfo& mPIInfo
    );

    void timeEvolutionE(
        thrust::device_vector<ElectricField>& E, 
        thrust::device_vector<MagneticField>& B, 
        thrust::device_vector<CurrentField>& current, 
        const double dt, 
        MPIInfo& mPIInfo
    );

private:

};


