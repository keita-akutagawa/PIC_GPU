#include "boundary.hpp"



//////////

void Boundary::periodicBoundaryBX(
    thrust::device_vector<MagneticField>& B
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}


void Boundary::periodicBoundaryBY(
    thrust::device_vector<MagneticField>& B
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}

//////////

void Boundary::periodicBoundaryEX(
    thrust::device_vector<ElectricField>& E
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}


void Boundary::periodicBoundaryEY(
    thrust::device_vector<ElectricField>& E
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}

//////////

void Boundary::periodicBoundaryCurrentX(
    thrust::device_vector<CurrentField>& current
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}


void Boundary::periodicBoundaryCurrentY(
    thrust::device_vector<CurrentField>& current
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}



