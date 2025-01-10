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
    int mpiBufNumParticles = 0; 

    unsigned long long existNumIonPerProcs = 0;
    unsigned long long existNumElectronPerProcs = 0;
    unsigned long long totalNumIonPerProcs = 0;
    unsigned long long totalNumElectronPerProcs = 0;

    float xminForProcs = 0.0f;
    float xmaxForProcs = 0.0f;
    float yminForProcs = 0.0f;
    float ymaxForProcs = 0.0f; 

    unsigned int numForSendParticlesIonLeft = 0;
    unsigned int numForSendParticlesIonRight = 0;
    unsigned int numForRecvParticlesIonLeft = 0;
    unsigned int numForRecvParticlesIonRight = 0;

    unsigned int numForSendParticlesIonCornerLeftDown = 0;
    unsigned int numForSendParticlesIonCornerRightDown = 0;
    unsigned int numForSendParticlesIonCornerLeftUp = 0;
    unsigned int numForSendParticlesIonCornerRightUp = 0;
    unsigned int numForRecvParticlesIonCornerLeftDown = 0;
    unsigned int numForRecvParticlesIonCornerRightDown = 0;
    unsigned int numForRecvParticlesIonCornerLeftUp = 0;
    unsigned int numForRecvParticlesIonCornerRightUp = 0;

    unsigned int numForSendParticlesIonDown = 0;
    unsigned int numForSendParticlesIonUp = 0;
    unsigned int numForRecvParticlesIonDown = 0;
    unsigned int numForRecvParticlesIonUp = 0;


    unsigned int numForSendParticlesElectronLeft = 0;
    unsigned int numForSendParticlesElectronRight = 0;
    unsigned int numForRecvParticlesElectronLeft = 0;
    unsigned int numForRecvParticlesElectronRight = 0;

    unsigned int numForSendParticlesElectronCornerLeftDown = 0;
    unsigned int numForSendParticlesElectronCornerRightDown = 0;
    unsigned int numForSendParticlesElectronCornerLeftUp = 0;
    unsigned int numForSendParticlesElectronCornerRightUp = 0;
    unsigned int numForRecvParticlesElectronCornerLeftDown = 0;
    unsigned int numForRecvParticlesElectronCornerRightDown = 0;
    unsigned int numForRecvParticlesElectronCornerLeftUp = 0;
    unsigned int numForRecvParticlesElectronCornerRightUp = 0;

    unsigned int numForSendParticlesElectronDown = 0;
    unsigned int numForSendParticlesElectronUp = 0;
    unsigned int numForRecvParticlesElectronDown = 0;
    unsigned int numForRecvParticlesElectronUp = 0;

    MPI_Datatype mpi_particle_type;
    MPI_Datatype mpi_field_type;


    __host__ __device__
    int getRank(int dx, int dy);

    __host__ __device__
    bool isInside(int globalX, int globalY);

    __host__ __device__
    int globalToLocal(int globalX, int globalY);
};


void setupInfo(MPIInfo& mPIInfo, int buffer, int mpiBufNumParticles);


template <typename FieldType>
void sendrecv_field_x(
    thrust::device_vector<FieldType>& field, 
    thrust::device_vector<FieldType>& sendFieldLeft, 
    thrust::device_vector<FieldType>& sendFieldRight, 
    thrust::device_vector<FieldType>& recvFieldLeft, 
    thrust::device_vector<FieldType>& recvFieldRight, 
    MPIInfo& mPIInfo)
{
    int localNx = mPIInfo.localNx;
    int localNy = mPIInfo.localNy;
    //int localSizeX = mPIInfo.localSizeX;
    int localSizeY = mPIInfo.localSizeY;

    int left  = mPIInfo.getRank(-1, 0);
    int right = mPIInfo.getRank(1, 0);
    MPI_Status st;

    FieldType* d_field = thrust::raw_pointer_cast(field.data());
    FieldType* d_sendFieldLeft = thrust::raw_pointer_cast(sendFieldLeft.data());
    FieldType* d_sendFieldRight = thrust::raw_pointer_cast(sendFieldRight.data());
    for (int i = 0; i < mPIInfo.buffer; i++) {
        cudaMemcpy(
            d_sendFieldLeft + i * localNy, 
            d_field + (mPIInfo.buffer + i) * localSizeY + mPIInfo.buffer,
            localNy * sizeof(FieldType),
            cudaMemcpyDeviceToDevice
        );
        cudaMemcpy(
            d_sendFieldRight + i * localNy,
            d_field + (localNx + i) * localSizeY + mPIInfo.buffer,
            localNy * sizeof(FieldType),
            cudaMemcpyDeviceToDevice
        );
    }

    MPI_Sendrecv(thrust::raw_pointer_cast(sendFieldLeft.data()),  sendFieldLeft.size(),  mPIInfo.mpi_field_type, left,  0, 
                 thrust::raw_pointer_cast(recvFieldRight.data()), recvFieldRight.size(), mPIInfo.mpi_field_type, right, 0, 
                 MPI_COMM_WORLD, &st);
    MPI_Sendrecv(thrust::raw_pointer_cast(sendFieldRight.data()), sendFieldRight.size(), mPIInfo.mpi_field_type, right, 0, 
                 thrust::raw_pointer_cast(recvFieldLeft.data()),  recvFieldLeft.size(),  mPIInfo.mpi_field_type, left,  0, 
                 MPI_COMM_WORLD, &st);

    FieldType* d_recvFieldLeft = thrust::raw_pointer_cast(recvFieldLeft.data());
    FieldType* d_recvFieldRight = thrust::raw_pointer_cast(recvFieldRight.data());
    for (int i = 0; i < mPIInfo.buffer; i++) {
        cudaMemcpy(
            d_field + i * localSizeY + mPIInfo.buffer,
            d_recvFieldLeft + i * localNy,
            localNy * sizeof(FieldType),
            cudaMemcpyDeviceToDevice
        ); 
        cudaMemcpy(
            d_field + (localNx + mPIInfo.buffer + i) * localSizeY + mPIInfo.buffer,
            d_recvFieldRight + i * localNy,
            localNy * sizeof(FieldType),
            cudaMemcpyDeviceToDevice
        ); 
    }
}


template <typename FieldType>
void sendrecv_field_y(
    thrust::device_vector<FieldType>& field, 
    thrust::device_vector<FieldType>& sendFieldUp, 
    thrust::device_vector<FieldType>& sendFieldDown, 
    thrust::device_vector<FieldType>& recvFieldUp, 
    thrust::device_vector<FieldType>& recvFieldDown, 
    MPIInfo& mPIInfo)
{
    //int localNx = mPIInfo.localNx;
    int localNy = mPIInfo.localNy;
    int localSizeX = mPIInfo.localSizeX;
    int localSizeY = mPIInfo.localSizeY;

    int down = mPIInfo.getRank(0, -1);
    int up   = mPIInfo.getRank(0, 1);   
    MPI_Status st;

    FieldType* d_field = thrust::raw_pointer_cast(field.data());
    FieldType* d_sendFieldDown = thrust::raw_pointer_cast(sendFieldDown.data());
    FieldType* d_sendFieldUp = thrust::raw_pointer_cast(sendFieldUp.data());
    for (int i = 0; i < localSizeX; i++) {
        cudaMemcpy(
            d_sendFieldDown + i * mPIInfo.buffer,
            d_field + mPIInfo.buffer + i * localSizeY,
            mPIInfo.buffer * sizeof(FieldType),
            cudaMemcpyDeviceToDevice
        );
        cudaMemcpy(
            d_sendFieldUp + i * mPIInfo.buffer,
            d_field + localNy + i * localSizeY,
            mPIInfo.buffer * sizeof(FieldType),
            cudaMemcpyDeviceToDevice
        );
    }

    MPI_Sendrecv(thrust::raw_pointer_cast(sendFieldDown.data()), sendFieldDown.size(), mPIInfo.mpi_field_type, down, 0, 
                 thrust::raw_pointer_cast(recvFieldUp.data()),   recvFieldUp.size(),   mPIInfo.mpi_field_type, up,   0, 
                 MPI_COMM_WORLD, &st);
    MPI_Sendrecv(thrust::raw_pointer_cast(sendFieldUp.data()),   sendFieldUp.size(),   mPIInfo.mpi_field_type, up,   0, 
                 thrust::raw_pointer_cast(recvFieldDown.data()), recvFieldDown.size(), mPIInfo.mpi_field_type, down, 0, 
                 MPI_COMM_WORLD, &st);

    FieldType* d_recvFieldDown = thrust::raw_pointer_cast(recvFieldDown.data());
    FieldType* d_recvFieldUp = thrust::raw_pointer_cast(recvFieldUp.data());
    for (int i = 0; i < localSizeX; i++) {
        cudaMemcpy(
            d_field + i * localSizeY,
            d_recvFieldDown + i * mPIInfo.buffer,
            mPIInfo.buffer * sizeof(FieldType),
            cudaMemcpyDeviceToDevice
        );
        cudaMemcpy(
            d_field + localNy + mPIInfo.buffer + i * localSizeY,
            d_recvFieldUp + i * mPIInfo.buffer,
            mPIInfo.buffer * sizeof(FieldType),
            cudaMemcpyDeviceToDevice
        );
    }
}

void sendrecv_magneticField_x(
    thrust::device_vector<MagneticField>& B, 
    thrust::device_vector<MagneticField>& sendMagneticFieldLeft, 
    thrust::device_vector<MagneticField>& sendMagneticFieldRight, 
    thrust::device_vector<MagneticField>& recvMagneticFieldLeft, 
    thrust::device_vector<MagneticField>& recvMagneticFieldRight, 
    MPIInfo& mPIInfo
); 

void sendrecv_magneticField_y(
    thrust::device_vector<MagneticField>& B, 
    thrust::device_vector<MagneticField>& sendMagneticFieldDown, 
    thrust::device_vector<MagneticField>& sendMagneticFieldUp, 
    thrust::device_vector<MagneticField>& recvMagneticFieldDown, 
    thrust::device_vector<MagneticField>& recvMagneticFieldUp, 
    MPIInfo& mPIInfo
); 


void sendrecv_electricField_x(
    thrust::device_vector<ElectricField>& E, 
    thrust::device_vector<ElectricField>& sendElectricFieldLeft, 
    thrust::device_vector<ElectricField>& sendElectricFieldRight, 
    thrust::device_vector<ElectricField>& recvElectricFieldLeft, 
    thrust::device_vector<ElectricField>& recvElectricFieldRight, 
    MPIInfo& mPIInfo
); 

void sendrecv_electricField_y(
    thrust::device_vector<ElectricField>& E, 
    thrust::device_vector<ElectricField>& sendElectricFieldDown, 
    thrust::device_vector<ElectricField>& sendElectricFieldUp, 
    thrust::device_vector<ElectricField>& recvElectricFieldDown, 
    thrust::device_vector<ElectricField>& recvElectricFieldUp, 
    MPIInfo& mPIInfo
); 


void sendrecv_currentField_x(
    thrust::device_vector<CurrentField>& current, 
    thrust::device_vector<CurrentField>& sendCurrentFieldLeft, 
    thrust::device_vector<CurrentField>& sendCurrentFieldRight, 
    thrust::device_vector<CurrentField>& recvCurrentFieldLeft, 
    thrust::device_vector<CurrentField>& recvCurrentFieldRight, 
    MPIInfo& mPIInfo
); 

void sendrecv_currentField_y(
    thrust::device_vector<CurrentField>& current, 
    thrust::device_vector<CurrentField>& sendCurrentFieldDown, 
    thrust::device_vector<CurrentField>& sendCurrentFieldUp, 
    thrust::device_vector<CurrentField>& recvCurrentFieldDown, 
    thrust::device_vector<CurrentField>& recvCurrentFieldUp, 
    MPIInfo& mPIInfo
); 


void sendrecv_numParticle_x(
    const unsigned int& numForSendParticlesSpeciesLeft, 
    const unsigned int& numForSendParticlesSpeciesRight, 
    unsigned int& numForRecvParticlesSpeciesLeft, 
    unsigned int& numForRecvParticlesSpeciesRight, 
    MPIInfo& mPIInfo
);

void sendrecv_numParticle_corner(
    const unsigned int& numForSendParticlesSpeciesCornerLeftDown, 
    const unsigned int& numForSendParticlesSpeciesCornerRightDown, 
    const unsigned int& numForSendParticlesSpeciesCornerLeftUp, 
    const unsigned int& numForSendParticlesSpeciesCornerRightUp, 
    unsigned int& numForRecvParticlesSpeciesCornerLeftDown, 
    unsigned int& numForRecvParticlesSpeciesCornerRightDown, 
    unsigned int& numForRecvParticlesSpeciesDown, 
    unsigned int& numForRecvParticlesSpeciesUp, 
    MPIInfo& mPIInfo
);

void sendrecv_numParticle_y(
    const unsigned int& numForSendParticlesSpeciesDown, 
    const unsigned int& numForSendParticlesSpeciesUp, 
    unsigned int& numForRecvParticlesSpeciesDown, 
    unsigned int& numForRecvParticlesSpeciesUp, 
    MPIInfo& mPIInfo
);


void sendrecv_particle_x(
    thrust::device_vector<Particle>& sendParticlesSpeciesLeft,
    thrust::device_vector<Particle>& sendParticlesSpeciesRight,  
    thrust::device_vector<Particle>& recvParticlesSpeciesLeft,
    thrust::device_vector<Particle>& recvParticlesSpeciesRight,
    const unsigned int& numForSendParticlesSpeciesLeft, 
    const unsigned int& numForSendParticlesSpeciesRight, 
    const unsigned int& numForRecvParticlesSpeciesLeft, 
    const unsigned int& numForRecvParticlesSpeciesRight, 
    MPIInfo& mPIInfo
);

void sendrecv_particle_y(
    thrust::device_vector<Particle>& sendParticlesSpeciesDown,
    thrust::device_vector<Particle>& sendParticlesSpeciesUp,  
    thrust::device_vector<Particle>& recvParticlesSpeciesDown,
    thrust::device_vector<Particle>& recvParticlesSpeciesUp, 
    const unsigned int& numForSendParticlesSpeciesDown, 
    const unsigned int& numForSendParticlesSpeciesUp, 
    const unsigned int& numForRecvParticlesSpeciesDown, 
    const unsigned int& numForRecvParticlesSpeciesUp, 
    MPIInfo& mPIInfo
);

#endif


