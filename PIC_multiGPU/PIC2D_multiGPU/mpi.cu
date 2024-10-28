#include "mpi.hpp"


int MPIInfo::getRank(int dx, int dy)
{
    int rankX = (localGridX + dx + gridX) % gridX;
    int rankY = (localGridY + dy + gridY) % gridY;
    return rankY + rankX * gridY;
}


bool MPIInfo::isInside(int globalX, int globalY)
{
    int startX = localNx * localGridX;
    int endX = startX + localNx;
    int startY = localNy * localGridY;
    int endY = startY + localNy;

    if (globalX < startX) return false;
    if (globalX >= endX) return false;
    if (globalY < startY) return false;
    if (globalY >= endY) return false;

    return true;
}


int MPIInfo::globalToLocal(int globalX, int globalY)
{
    int startX = localNx * localGridX;
    int x = globalX - startX;

    int startY = localNy * localGridY;
    int y = globalY - startY;

    return y + buffer + (x + buffer) * localSizeY;
}


void setupInfo(MPIInfo& mPIInfo, int buffer)
{
    int rank = 0, procs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    int d2[2] = {};
    MPI_Dims_create(procs, 2, d2);
    mPIInfo.rank = rank;
    mPIInfo.procs = procs;
    mPIInfo.gridX = d2[0];
    mPIInfo.gridY = d2[1];
    mPIInfo.localGridX = rank / mPIInfo.gridX;
    mPIInfo.localGridY = rank % mPIInfo.gridX;
    mPIInfo.localNx = nx / mPIInfo.gridX;
    mPIInfo.localNy = ny / mPIInfo.gridY;
    mPIInfo.buffer = buffer;
    mPIInfo.localSizeX = mPIInfo.localNx + 2 * mPIInfo.buffer;
    mPIInfo.localSizeY = mPIInfo.localNy + 2 * mPIInfo.buffer;


    int block_lengths_particle[12] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    MPI_Aint offsets_particle[12];
    offsets_particle[0]  = offsetof(Particle, x);
    offsets_particle[1]  = offsetof(Particle, y);
    offsets_particle[2]  = offsetof(Particle, z);
    offsets_particle[3]  = offsetof(Particle, vx);
    offsets_particle[4]  = offsetof(Particle, vy);
    offsets_particle[5]  = offsetof(Particle, vz);
    offsets_particle[6]  = offsetof(Particle, gamma);
    offsets_particle[7]  = offsetof(Particle, isExist);
    offsets_particle[8]  = offsetof(Particle, isMPISendLeftToRight);
    offsets_particle[9]  = offsetof(Particle, isMPISendRightToLeft);
    offsets_particle[10] = offsetof(Particle, isMPISendUpToDown);
    offsets_particle[11] = offsetof(Particle, isMPISendDownToUp);

    MPI_Datatype types_particle[12] = {
        MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, 
        MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, 
        MPI_FLOAT, MPI_C_BOOL, 
        MPI_C_BOOL, MPI_C_BOOL, MPI_C_BOOL, MPI_C_BOOL
    };

    MPI_Type_create_struct(12, block_lengths_particle, offsets_particle, types_particle, &mPIInfo.mpi_particle_type);
    MPI_Type_commit(&mPIInfo.mpi_particle_type);

    // MagneticField, ElectricField, CurrentField共通
    int block_lengths_field[3] = {1, 1, 1};
    MPI_Aint offsets_field[3];
    offsets_field[0] = offsetof(MagneticField, bX);
    offsets_field[1] = offsetof(MagneticField, bY);
    offsets_field[2] = offsetof(MagneticField, bZ);
    MPI_Datatype types_field[3] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
    MPI_Type_create_struct(3, block_lengths_field, offsets_field, types_field, &mPIInfo.mpi_field_type);
    MPI_Type_commit(&mPIInfo.mpi_field_type);
}


//////////////////////////////////////////////////

// sendrecv(field用)はヘッダーファイルにある。
// templeteを使用したため


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
)
{
    int left = mPIInfo.getRank(-1, 0);
    int right = mPIInfo.getRank(1, 0);
    MPI_Status st;

    MPI_Sendrecv(
        &(countForSendSpeciesRightToLeft), 
        1, 
        MPI_UNSIGNED_LONG_LONG, 
        right, 0, 
        &(countForRecvSpeciesRightToLeft), 
        1, 
        MPI_UNSIGNED_LONG_LONG, 
        left, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        &(countForSendSpeciesLeftToRight), 
        1, 
        MPI_UNSIGNED_LONG_LONG,  
        left, 0, 
        &(countForRecvSpeciesLeftToRight), 
        1, 
        MPI_UNSIGNED_LONG_LONG, 
        right, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        thrust::raw_pointer_cast(sendParticlesSpeciesRightToLeft.data()), 
        countForSendSpeciesRightToLeft, 
        mPIInfo.mpi_particle_type, 
        right, 0, 
        thrust::raw_pointer_cast(recvParticlesSpeciesRightToLeft.data()),
        countForRecvSpeciesRightToLeft, 
        mPIInfo.mpi_particle_type, 
        left, 0, 
        MPI_COMM_WORLD, &st
    );
    
    MPI_Sendrecv(
        thrust::raw_pointer_cast(sendParticlesSpeciesLeftToRight.data()), 
        countForSendSpeciesLeftToRight, 
        mPIInfo.mpi_particle_type, 
        left, 0, 
        thrust::raw_pointer_cast(recvParticlesSpeciesLeftToRight.data()), 
        countForRecvSpeciesLeftToRight,   
        mPIInfo.mpi_particle_type, 
        right, 0, 
        MPI_COMM_WORLD, &st
    );
}


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
)
{
    int up = mPIInfo.getRank(0, -1);
    int down = mPIInfo.getRank(0, 1);
    MPI_Status st;

    MPI_Sendrecv(
        &(countForSendSpeciesDownToUp), 
        1, 
        MPI_UNSIGNED_LONG_LONG, 
        down, 0, 
        &(countForRecvSpeciesDownToUp), 
        1, 
        MPI_UNSIGNED_LONG_LONG, 
        up, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        &(countForSendSpeciesUpToDown), 
        1, 
        MPI_UNSIGNED_LONG_LONG,  
        up, 0, 
        &(countForRecvSpeciesUpToDown), 
        1, 
        MPI_UNSIGNED_LONG_LONG, 
        down, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        thrust::raw_pointer_cast(sendParticlesSpeciesDownToUp.data()), 
        countForSendSpeciesDownToUp, 
        mPIInfo.mpi_particle_type, 
        down, 0, 
        thrust::raw_pointer_cast(recvParticlesSpeciesDownToUp.data()), 
        countForRecvSpeciesDownToUp, 
        mPIInfo.mpi_particle_type, 
        up, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        thrust::raw_pointer_cast(sendParticlesSpeciesUpToDown.data()), 
        countForSendSpeciesUpToDown,
        mPIInfo.mpi_particle_type, 
        up, 0, 
        thrust::raw_pointer_cast(recvParticlesSpeciesUpToDown.data()), 
        countForRecvSpeciesUpToDown,  
        mPIInfo.mpi_particle_type, 
        down, 0, 
        MPI_COMM_WORLD, &st
    );
}

