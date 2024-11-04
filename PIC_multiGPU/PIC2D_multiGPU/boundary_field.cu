#include "boundary.hpp"



//////////

void Boundary::periodicBoundaryB_x(
    thrust::device_vector<MagneticField>& B
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}


void Boundary::periodicBoundaryB_y(
    thrust::device_vector<MagneticField>& B
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}

//////////

void Boundary::periodicBoundaryE_x(
    thrust::device_vector<ElectricField>& E
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}


void Boundary::periodicBoundaryE_y(
    thrust::device_vector<ElectricField>& E
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}

//////////

void Boundary::periodicBoundaryCurrent_x(
    thrust::device_vector<CurrentField>& current
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}


void Boundary::periodicBoundaryCurrent_y(
    thrust::device_vector<CurrentField>& current
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}



