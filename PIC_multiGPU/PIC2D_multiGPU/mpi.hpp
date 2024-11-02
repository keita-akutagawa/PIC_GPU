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
    int rank = 0;
    int procs = 0;
    int gridX, gridY = 0;
    int localGridX, localGridY = 0;
    int localNx = 0, localNy = 0; 
    int buffer = 0;
    int localSizeX = 0, localSizeY = 0;

    unsigned long long existNumIonPerProcs = 0;
    unsigned long long existNumElectronPerProcs = 0;
    unsigned long long totalNumIonPerProcs = 0;
    unsigned long long totalNumElectronPerProcs = 0;

    float xminForProcs = 0.0f;
    float xmaxForProcs = 0.0f;
    float yminForProcs = 0.0f;
    float ymaxForProcs = 0.0f; 

    unsigned int numForSendParticlesIonLeftward = 0;
    unsigned int numForSendParticlesIonRightward = 0;
    unsigned int numForRecvParticlesIonLeftward = 0;
    unsigned int numForRecvParticlesIonRightward = 0;

    unsigned int numForSendParticlesIonDownward = 0;
    unsigned int numForSendParticlesIonUpward = 0;
    unsigned int numForRecvParticlesIonDownward = 0;
    unsigned int numForRecvParticlesIonUpward = 0;

    unsigned int numForSendParticlesElectronLeftward = 0;
    unsigned int numForSendParticlesElectronRightward = 0;
    unsigned int numForRecvParticlesElectronLeftward = 0;
    unsigned int numForRecvParticlesElectronRightward = 0;

    unsigned int numForSendParticlesElectronDownward = 0;
    unsigned int numForSendParticlesElectronUpward = 0;
    unsigned int numForRecvParticlesElectronDownward = 0;
    unsigned int numForRecvParticlesElectronUpward = 0;

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
            sendFieldLeft[ j + i * localNy] = field[j + mPIInfo.buffer + (mPIInfo.buffer + i) * localSizeY];
            sendFieldRight[j + i * localNy] = field[j + mPIInfo.buffer + (localNx + i)        * localSizeY];
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


void sendrecv_num_particle_x(
    const unsigned int& numForSendParticlesSpeciesLeft, 
    const unsigned int& numForSendParticlesSpeciesRight, 
    unsigned int& numForRecvParticlesSpeciesLeft, 
    unsigned int& numForRecvParticlesSpeciesRight, 
    MPIInfo& mPIInfo
);

void sendrecv_num_particle_y(
    const unsigned int& numForSendParticlesSpeciesDown, 
    const unsigned int& numForSendParticlesSpeciesUp, 
    unsigned int& numForRecvParticlesSpeciesDown, 
    unsigned int& numForRecvParticlesSpeciesUp, 
    MPIInfo& mPIInfo
);


void sendrecv_particle_x(
    thrust::host_vector<Particle>& host_sendParticlesSpeciesLeft,
    thrust::host_vector<Particle>& host_sendParticlesSpeciesRight,  
    thrust::host_vector<Particle>& host_recvParticlesSpeciesLeft,
    thrust::host_vector<Particle>& host_recvParticlesSpeciesRight,
    MPIInfo& mPIInfo
);

void sendrecv_particle_y(
    thrust::host_vector<Particle>& host_sendParticlesSpeciesDown,
    thrust::host_vector<Particle>& host_sendParticlesSpeciesUp,  
    thrust::host_vector<Particle>& host_recvParticlesSpeciesDown,
    thrust::host_vector<Particle>& host_recvParticlesSpeciesUp,
    MPIInfo& mPIInfo
);

#endif


