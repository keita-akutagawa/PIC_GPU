#ifndef MPI_H
#define MPI_H

#include <thrust/device_vector.h>
#include <mpi.h>
#include "const.hpp"


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


    __host__ __device__
    int getRank(int dx);

    __host__ __device__
    bool isInside(int globalX);

    __host__ __device__
    int globalToLocal(int globalX);
};


void setupInfo(MPIInfo& mPIInfo);


template <typename FieldType>
void sendrecv(thrust::device_vector<FieldType>& field, MPIInfo& mPIInfo)
{
    int localNx = mPIInfo.localNx;
    int left = mPIInfo.getRank(-1);
    int right = mPIInfo.getRank(1);
    MPI_Status st;

    FieldType sendFieldLeft, sendFieldRight;
    FieldType recvFieldLeft, recvFieldRight;

    sendFieldRight = field[localNx];
    sendFieldLeft  = field[1];

    MPI_Sendrecv(&(sendFieldRight), 3, MPI_FLOAT, right, 0, 
                 &(recvFieldLeft),  3, MPI_FLOAT, left,  0, MPI_COMM_WORLD, &st
    );
    MPI_Sendrecv(&(sendFieldLeft),  3, MPI_FLOAT, right, 0, 
                 &(recvFieldRight), 3, MPI_FLOAT, left,  0, MPI_COMM_WORLD, &st
    );

    field[0]           = recvFieldLeft;
    field[localNx + 1] = recvFieldRight;
}


void sendrecv_particle(thrust::device_vector<Particle>& particlesSpecies, MPIInfo& mPIInfo)
{
    
}

#endif


