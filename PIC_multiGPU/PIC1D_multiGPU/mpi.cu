#include "mpi.hpp"


int MPIInfo::getRank(int dx)
{
    int rankX = (localGridX + dx + gridX) % gridX;
    return rankX;
}


bool MPIInfo::isInside(int globalX)
{
    int startX = localNx * localGridX;
    int endX = startX + localNx;

    if (globalX < startX) return false;
    if (globalX >= endX) return false;

    return true;
}


int MPIInfo::globalToLocal(int globalX)
{
    int startX = localNx * localGridX;

    int x = globalX - startX;

    return x + 1;
}


void setupInfo(MPIInfo& mPIInfo)
{
    int rank = 0, procs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    mPIInfo.rank = rank;
    mPIInfo.procs = procs;
    mPIInfo.gridX = procs;
    mPIInfo.localGridX = rank;
    mPIInfo.localNx = nx / mPIInfo.gridX;


    int block_lengths_particle[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    MPI_Aint offsets_particle[8];
    offsets_particle[0] = offsetof(Particle, x);
    offsets_particle[1] = offsetof(Particle, y);
    offsets_particle[2] = offsetof(Particle, z);
    offsets_particle[3] = offsetof(Particle, vx);
    offsets_particle[4] = offsetof(Particle, vy);
    offsets_particle[5] = offsetof(Particle, vz);
    offsets_particle[6] = offsetof(Particle, gamma);
    offsets_particle[7] = offsetof(Particle, isExist);

    MPI_Datatype types_particle[8] = {
        MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, 
        MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, 
        MPI_DOUBLE, MPI_C_BOOL
    };

    MPI_Type_create_struct(8, block_lengths_particle, offsets_particle, types_particle, &mPIInfo.mpi_particle_type);
    MPI_Type_commit(&mPIInfo.mpi_particle_type);

    // MagneticField, ElectricField, CurrentField共通
    int block_lengths_field[3] = {1, 1, 1};
    MPI_Aint offsets_field[3];
    offsets_field[0] = offsetof(MagneticField, bX);
    offsets_field[1] = offsetof(MagneticField, bY);
    offsets_field[2] = offsetof(MagneticField, bZ);
    MPI_Datatype types_field[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Type_create_struct(3, block_lengths_field, offsets_field, types_field, &mPIInfo.mpi_field_type);
    MPI_Type_commit(&mPIInfo.mpi_field_type);
}


//////////////////////////////////////////////////

// sendrecv(field用)はヘッダーファイルにある。
// templeteを使用したため


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
)
{
    int left = mPIInfo.getRank(-1);
    int right = mPIInfo.getRank(1);
    MPI_Status st;


    MPI_Sendrecv(
        host_sendParticlesSpeciesRightToLeft.data(), 
        host_sendParticlesSpeciesRightToLeft.size(), 
        mPIInfo.mpi_particle_type, 
        right, 0, 
        host_recvParticlesSpeciesRightToLeft.data(), 
        host_recvParticlesSpeciesRightToLeft.size(), 
        mPIInfo.mpi_particle_type, 
        left, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        host_sendParticlesSpeciesLeftToRight.data(), 
        host_sendParticlesSpeciesLeftToRight.size(),
        mPIInfo.mpi_particle_type, 
        left, 0, 
        host_recvParticlesSpeciesLeftToRight.data(), 
        host_recvParticlesSpeciesLeftToRight.size(),  
        mPIInfo.mpi_particle_type, 
        right, 0, 
        MPI_COMM_WORLD, &st
    );


    MPI_Sendrecv(
        &(countForSendSpeciesRightToLeft), 
        1, 
        MPI_INT, 
        right, 0, 
        &(countForRecvSpeciesRightToLeft), 
        1, 
        MPI_INT, 
        left, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        &(countForSendSpeciesLeftToRight), 
        1, 
        MPI_INT,  
        left, 0, 
        &(countForRecvSpeciesLeftToRight), 
        1, 
        MPI_INT, 
        right, 0, 
        MPI_COMM_WORLD, &st
    );
}


