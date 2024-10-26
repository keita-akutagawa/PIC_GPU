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
    int gridX, gridY;
    int localGridX, localGridY;
    int localNx, localNy; 
    int buffer;
    int localSizeX, localSizeY; 
    
    unsigned long long existNumIonPerProcs = 0;
    unsigned long long existNumElectronPerProcs = 0;
    unsigned long long totalNumIonPerProcs = 0;
    unsigned long long totalNumElectronPerProcs = 0;

    MPI_Datatype mpi_particle_type;
    MPI_Datatype mpi_field_type;


    __host__ __device__
    int getRank(int dx, int dy);

    __host__ __device__
    bool isInside(int globalX, int globalY);

    __host__ __device__
    int globalToLocal(int globalX, int globalY);
};


void setupInfo(MPIInfo& mPIInfo, int buffer);


template <typename FieldType>
void sendrecv_field_x(thrust::device_vector<FieldType>& field, MPIInfo& mPIInfo)
{
    int localNx = mPIInfo.localNx;
    int localNy = mPIInfo.localNy;
    //int localSizeX = mPIInfo.localSizeX;
    int localSizeY = mPIInfo.localSizeY;

    int left = mPIInfo.getRank(-1, 0);
    int right = mPIInfo.getRank(1, 0);
    MPI_Status st;

    thrust::host_vector<FieldType> sendFieldLeft(mPIInfo.buffer * localNy), sendFieldRight(mPIInfo.buffer * localNy);
    thrust::host_vector<FieldType> recvFieldLeft(mPIInfo.buffer * localNy), recvFieldRight(mPIInfo.buffer * localNy);

    for (int i = 0; i < mPIInfo.buffer; i++) {
        for (int j = 0; j < localNy; j++) {
            sendFieldRight[j + i * localNy] = field[j + mPIInfo.buffer + (localNx + i)        * localSizeY];
            sendFieldLeft[ j + i * localNy] = field[j + mPIInfo.buffer + (mPIInfo.buffer + i) * localSizeY];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Sendrecv(sendFieldRight.data(), sendFieldRight.size(), mPIInfo.mpi_field_type, right, 0, 
                 recvFieldLeft.data(),  recvFieldLeft.size(),  mPIInfo.mpi_field_type, left,  0, 
                 MPI_COMM_WORLD, &st);
    MPI_Sendrecv(sendFieldLeft.data(),  sendFieldLeft.size(),  mPIInfo.mpi_field_type, left,  0, 
                 recvFieldRight.data(), recvFieldRight.size(), mPIInfo.mpi_field_type, right, 0, 
                 MPI_COMM_WORLD, &st);
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < mPIInfo.buffer; i++) {
        for (int j = 0; j < localNy; j++) {
            field[j + mPIInfo.buffer + i                              * localSizeY] = recvFieldLeft[ j + i * localNy];
            field[j + mPIInfo.buffer + (localNx + mPIInfo.buffer + i) * localSizeY] = recvFieldRight[j + i * localNy];
        }
    }
}


template <typename FieldType>
void sendrecv_field_y(thrust::device_vector<FieldType>& field, MPIInfo& mPIInfo)
{
    //int localNx = mPIInfo.localNx;
    int localNy = mPIInfo.localNy;
    int localSizeX = mPIInfo.localSizeX;
    int localSizeY = mPIInfo.localSizeY;

    int up = mPIInfo.getRank(0, -1);
    int down = mPIInfo.getRank(0, 1);
    MPI_Status st;

    thrust::host_vector<FieldType> sendFieldUp(mPIInfo.buffer * localSizeX), sendFieldDown(mPIInfo.buffer * localSizeX);
    thrust::host_vector<FieldType> recvFieldUp(mPIInfo.buffer * localSizeX), recvFieldDown(mPIInfo.buffer * localSizeX);

    for (int i = 0; i < localSizeX; i++) {
        for (int j = 0; j < mPIInfo.buffer; j++) {
            sendFieldDown[j + i * mPIInfo.buffer] = field[j + localNy        + i * localSizeY];
            sendFieldUp[  j + i * mPIInfo.buffer] = field[j + mPIInfo.buffer + i * localSizeY];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Sendrecv(sendFieldDown.data(), sendFieldDown.size(), mPIInfo.mpi_field_type, down, 0, 
                 recvFieldUp.data(),   recvFieldUp.size(),   mPIInfo.mpi_field_type, up,   0, 
                 MPI_COMM_WORLD, &st);
    MPI_Sendrecv(sendFieldUp.data(),   sendFieldUp.size(),   mPIInfo.mpi_field_type, up,   0, 
                 recvFieldDown.data(), recvFieldDown.size(), mPIInfo.mpi_field_type, down, 0, 
                 MPI_COMM_WORLD, &st);
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < localSizeX; i++) {
        for (int j = 0; j < mPIInfo.buffer; j++) {
            field[j                            + i * localSizeY] = recvFieldUp[  j + i * mPIInfo.buffer];
            field[j + localNy + mPIInfo.buffer + i * localSizeY] = recvFieldDown[j + i * mPIInfo.buffer];
        }
    }
}


template <typename FieldType>
void sendrecv_field(thrust::device_vector<FieldType>& field, MPIInfo& mPIInfo)
{
    MPI_Barrier(MPI_COMM_WORLD);
    sendrecv_field_x(field, mPIInfo);
    MPI_Barrier(MPI_COMM_WORLD);
    sendrecv_field_y(field, mPIInfo);
    MPI_Barrier(MPI_COMM_WORLD);
}


void sendrecv_particle_x(
    thrust::host_vector<Particle>& host_sendParticlesSpeciesLeftToRight,
    thrust::host_vector<Particle>& host_sendParticlesSpeciesRightToLeft,  
    thrust::host_vector<Particle>& host_recvParticlesSpeciesLeftToRight,
    thrust::host_vector<Particle>& host_recvParticlesSpeciesRightToLeft, 
    const unsigned long long& countForSendSpeciesLeftToRight, 
    const unsigned long long& countForSendSpeciesRightToLeft, 
    unsigned long long& countForRecvSpeciesLeftToRight, 
    unsigned long long& countForRecvSpeciesRightToLeft, 
    MPIInfo& mPIInfo
);


void sendrecv_particle_y(
    thrust::host_vector<Particle>& host_sendParticlesSpeciesUpToDown,
    thrust::host_vector<Particle>& host_sendParticlesSpeciesDownToUp,  
    thrust::host_vector<Particle>& host_recvParticlesSpeciesUpToDown,
    thrust::host_vector<Particle>& host_recvParticlesSpeciesDownToUp, 
    const unsigned long long& countForSendSpeciesUpToDown, 
    const unsigned long long& countForSendSpeciesDownToUp, 
    unsigned long long& countForRecvSpeciesUpToDown, 
    unsigned long long& countForRecvSpeciesDownToUp, 
    MPIInfo& mPIInfo
);


#endif


