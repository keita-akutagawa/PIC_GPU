#ifndef MPI_H
#define MPI_H

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <mpi.h>
#include "const.hpp"
#include "field_parameter_struct.hpp"
#include "particle_struct.hpp"


struct MPIInfo
{
    int rank;
    int procs;
    int gridX;
    int localGridX;
    int localNx; 

    int buffer = 1;
    int existNumIonPerProcs = 0;
    int existNumElectronPerProcs = 0;
    int totalNumIonPerProcs = 0;
    int totalNumElectronPerProcs = 0;

    MPI_Datatype mpi_particle_type;
    MPI_Datatype mpi_field_type;


    __host__ __device__
    int getRank(int dx);

    __host__ __device__
    bool isInside(int globalX);

    __host__ __device__
    int globalToLocal(int globalX);
};


void setupInfo(MPIInfo& mPIInfo);


template <typename FieldType>
void sendrecv_field(thrust::device_vector<FieldType>& field, MPIInfo& mPIInfo)
{
    int localNx = mPIInfo.localNx;
    int left = mPIInfo.getRank(-1);
    int right = mPIInfo.getRank(1);
    MPI_Status st;

    FieldType sendFieldLeft, sendFieldRight;
    FieldType recvFieldLeft, recvFieldRight;

    sendFieldRight = field[localNx];
    sendFieldLeft  = field[1];

    MPI_Sendrecv(&(sendFieldRight), 1, mPIInfo.mpi_field_type, right, 0, 
                 &(recvFieldLeft),  1, mPIInfo.mpi_field_type, left,  0, 
                 MPI_COMM_WORLD, &st
    );
    MPI_Sendrecv(&(sendFieldLeft),  1, mPIInfo.mpi_field_type, left,  0, 
                 &(recvFieldRight), 1, mPIInfo.mpi_field_type, right, 0, 
                 MPI_COMM_WORLD, &st
    );

    field[0]           = recvFieldLeft;
    field[localNx + 1] = recvFieldRight;
}


void sendrecv_particle(
    thrust::host_vector<Particle>& host_sendParticlesSpeciesLeftToRight,
    thrust::host_vector<Particle>& host_sendParticlesSpeciesRightToLeft,  
    thrust::host_vector<Particle>& host_recvParticlesSpeciesLeftToRight,
    thrust::host_vector<Particle>& host_recvParticlesSpeciesRightToLeft, 
    const int& countForSendSpeciesLeftToRight, 
    const int& countForSendSpeciesRightToLeft, 
    int& countForRecvSpeciesLeftToRight, 
    int& countForRecvSpeciesRightToLeft, 
    MPIInfo& mPIInfo
);


#endif


