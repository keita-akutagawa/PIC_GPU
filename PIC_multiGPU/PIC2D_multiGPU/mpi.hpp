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

    int left  = mPIInfo.getRank(-1, 0);
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

    MPI_Sendrecv(thrust::raw_pointer_cast(sendFieldRight.data()), sendFieldRight.size(), mPIInfo.mpi_field_type, right, 0, 
                 thrust::raw_pointer_cast(recvFieldLeft.data()),  recvFieldLeft.size(),  mPIInfo.mpi_field_type, left,  0, 
                 MPI_COMM_WORLD, &st);
    MPI_Sendrecv(thrust::raw_pointer_cast(sendFieldLeft.data()),  sendFieldLeft.size(),  mPIInfo.mpi_field_type, left,  0, 
                 thrust::raw_pointer_cast(recvFieldRight.data()), recvFieldRight.size(), mPIInfo.mpi_field_type, right, 0, 
                 MPI_COMM_WORLD, &st);

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

    int down = mPIInfo.getRank(0, -1);
    int up   = mPIInfo.getRank(0, 1);   
    MPI_Status st;

    thrust::host_vector<FieldType> sendFieldUp(mPIInfo.buffer * localSizeX), sendFieldDown(mPIInfo.buffer * localSizeX);
    thrust::host_vector<FieldType> recvFieldUp(mPIInfo.buffer * localSizeX), recvFieldDown(mPIInfo.buffer * localSizeX);

    for (int i = 0; i < localSizeX; i++) {
        for (int j = 0; j < mPIInfo.buffer; j++) {
            sendFieldUp[j + i * mPIInfo.buffer] = field[j + localNy        + i * localSizeY];
            sendFieldDown[  j + i * mPIInfo.buffer] = field[j + mPIInfo.buffer + i * localSizeY];
        }
    }

    MPI_Sendrecv(thrust::raw_pointer_cast(sendFieldUp.data()),   sendFieldUp.size(),   mPIInfo.mpi_field_type, up,   0, 
                 thrust::raw_pointer_cast(recvFieldDown.data()), recvFieldDown.size(), mPIInfo.mpi_field_type, down, 0, 
                 MPI_COMM_WORLD, &st);
    MPI_Sendrecv(thrust::raw_pointer_cast(sendFieldDown.data()), sendFieldDown.size(), mPIInfo.mpi_field_type, down, 0, 
                 thrust::raw_pointer_cast(recvFieldUp.data()),   recvFieldUp.size(),   mPIInfo.mpi_field_type, up,   0, 
                 MPI_COMM_WORLD, &st);

    for (int i = 0; i < localSizeX; i++) {
        for (int j = 0; j < mPIInfo.buffer; j++) {
            field[j                            + i * localSizeY] = recvFieldDown[  j + i * mPIInfo.buffer];
            field[j + localNy + mPIInfo.buffer + i * localSizeY] = recvFieldUp[j + i * mPIInfo.buffer];
        }
    }
}


template <typename FieldType>
void sendrecv_field(thrust::device_vector<FieldType>& field, MPIInfo& mPIInfo)
{
    sendrecv_field_x(field, mPIInfo);
    sendrecv_field_y(field, mPIInfo);
}


void sendrecv_particle_x(
    thrust::device_vector<Particle>& sendParticlesSpeciesLeftToRight,
    thrust::device_vector<Particle>& sendParticlesSpeciesRightToLeft,  
    thrust::device_vector<Particle>& recvParticlesSpeciesLeftToRight,
    thrust::device_vector<Particle>& recvParticlesSpeciesRightToLeft, 
    const unsigned long long& countForSendSpeciesLeftToRight, 
    const unsigned long long& countForSendSpeciesRightToLeft, 
    unsigned long long& countForRecvSpeciesLeftToRight, 
    unsigned long long& countForRecvSpeciesRightToLeft, 
    MPIInfo& mPIInfo
);


void sendrecv_particle_y(
    thrust::device_vector<Particle>& sendParticlesSpeciesUpToDown,
    thrust::device_vector<Particle>& sendParticlesSpeciesDownToUp,  
    thrust::device_vector<Particle>& recvParticlesSpeciesUpToDown,
    thrust::device_vector<Particle>& recvParticlesSpeciesDownToUp, 
    const unsigned long long& countForSendSpeciesUpToDown, 
    const unsigned long long& countForSendSpeciesDownToUp, 
    unsigned long long& countForRecvSpeciesUpToDown, 
    unsigned long long& countForRecvSpeciesDownToUp, 
    MPIInfo& mPIInfo
);


#endif


