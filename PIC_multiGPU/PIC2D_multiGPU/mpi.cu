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
    offsets_particle[8]  = offsetof(Particle, isMPISendLeftward);
    offsets_particle[9]  = offsetof(Particle, isMPISendRightward);
    offsets_particle[10] = offsetof(Particle, isMPISendUpward);
    offsets_particle[11] = offsetof(Particle, isMPISendDownward);

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


void sendrecv_num_particle_x(
    const unsigned int& numForSendParticlesSpeciesLeftward, 
    const unsigned int& numForSendParticlesSpeciesRightward, 
    unsigned int& numForRecvParticlesSpeciesLeftward, 
    unsigned int& numForRecvParticlesSpeciesRightward, 
    MPIInfo& mPIInfo
)
{
    int left  = mPIInfo.getRank(-1, 0);
    int right = mPIInfo.getRank(1, 0);
    MPI_Status st;

    MPI_Sendrecv(
        &(numForSendParticlesSpeciesRightward), 
        1, 
        MPI_UNSIGNED, 
        right, 0, 
        &(numForRecvParticlesSpeciesRightward), 
        1, 
        MPI_UNSIGNED, 
        left, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        &(numForSendParticlesSpeciesLeftward), 
        1, 
        MPI_UNSIGNED,  
        left, 0, 
        &(numForRecvParticlesSpeciesLeftward), 
        1, 
        MPI_UNSIGNED, 
        right, 0, 
        MPI_COMM_WORLD, &st
    );
}


void sendrecv_particle_x(
    thrust::host_vector<Particle>& host_sendParticlesSpeciesLeftward, 
    thrust::host_vector<Particle>& host_sendParticlesSpeciesRightward, 
    thrust::host_vector<Particle>& host_recvParticlesSpeciesLeftward, 
    thrust::host_vector<Particle>& host_recvParticlesSpeciesRightward, 
    MPIInfo& mPIInfo
)
{
    int left  = mPIInfo.getRank(-1, 0);
    int right = mPIInfo.getRank(1, 0);
    MPI_Status st;


    unsigned int maxNumLeftwardForProcs = max(
        host_sendParticlesSpeciesLeftward.size(), 
        host_recvParticlesSpeciesLeftward.size()
    );
    unsigned int maxNumRightwardForProcs = max(
        host_sendParticlesSpeciesRightward.size(), 
        host_recvParticlesSpeciesRightward.size()
    );

    unsigned int maxNumLeftward = 0, maxNumRightward = 0;
    MPI_Allreduce(&maxNumLeftwardForProcs, &maxNumLeftward, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&maxNumRightwardForProcs, &maxNumRightward, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
    
    thrust::host_vector<Particle> sendbufLeftward(maxNumLeftward);
    thrust::host_vector<Particle> sendbufRightward(maxNumRightward);
    thrust::host_vector<Particle> recvbufLeftward(maxNumLeftward);
    thrust::host_vector<Particle> recvbufRightward(maxNumRightward);

    for (unsigned int i = 0; i < host_sendParticlesSpeciesLeftward.size(); i++) {
        sendbufLeftward[i] = host_sendParticlesSpeciesLeftward[i];
    }
    for (unsigned int i = 0; i < host_sendParticlesSpeciesRightward.size(); i++) {
        sendbufRightward[i] = host_sendParticlesSpeciesRightward[i];
    }
    

    MPI_Sendrecv(
        thrust::raw_pointer_cast(sendbufRightward.data()), 
        sendbufRightward.size(), 
        mPIInfo.mpi_particle_type, 
        right, 0, 
        thrust::raw_pointer_cast(recvbufRightward.data()),
        recvbufRightward.size(), 
        mPIInfo.mpi_particle_type, 
        left, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        thrust::raw_pointer_cast(sendbufLeftward.data()), 
        sendbufLeftward.size(), 
        mPIInfo.mpi_particle_type, 
        left, 0, 
        thrust::raw_pointer_cast(recvbufLeftward.data()), 
        recvbufLeftward.size(), 
        mPIInfo.mpi_particle_type, 
        right, 0, 
        MPI_COMM_WORLD, &st
    );


    for (unsigned int i = 0; i < host_recvParticlesSpeciesLeftward.size(); i++) {
        host_recvParticlesSpeciesLeftward[i] = recvbufLeftward[i];
    }
    for (unsigned int i = 0; i < host_recvParticlesSpeciesRightward.size(); i++) {
        host_recvParticlesSpeciesRightward[i] = recvbufRightward[i];
    }
}


void sendrecv_num_particle_y(
    const unsigned int& numForSendParticlesSpeciesDownward, 
    const unsigned int& numForSendParticlesSpeciesUpward, 
    unsigned int& numForRecvParticlesSpeciesDownward, 
    unsigned int& numForRecvParticlesSpeciesUpward, 
    MPIInfo& mPIInfo
)
{
    int down = mPIInfo.getRank(0, -1);
    int up   = mPIInfo.getRank(0, 1);
    MPI_Status st;

    MPI_Sendrecv(
        &(numForSendParticlesSpeciesUpward), 
        1, 
        MPI_UNSIGNED, 
        up, 0, 
        &(numForRecvParticlesSpeciesUpward), 
        1, 
        MPI_UNSIGNED, 
        down, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        &(numForSendParticlesSpeciesDownward), 
        1, 
        MPI_UNSIGNED,  
        down, 0, 
        &(numForRecvParticlesSpeciesDownward), 
        1, 
        MPI_UNSIGNED, 
        up, 0, 
        MPI_COMM_WORLD, &st
    );
}


void sendrecv_particle_y(
    thrust::host_vector<Particle>& host_sendParticlesSpeciesDownward, 
    thrust::host_vector<Particle>& host_sendParticlesSpeciesUpward, 
    thrust::host_vector<Particle>& host_recvParticlesSpeciesDownward, 
    thrust::host_vector<Particle>& host_recvParticlesSpeciesUpward, 
    MPIInfo& mPIInfo
)
{
    int down  = mPIInfo.getRank(0, -1);
    int up = mPIInfo.getRank(0, 1);
    MPI_Status st;


    unsigned int maxNumDownwardForProcs = max(
        host_sendParticlesSpeciesDownward.size(), 
        host_recvParticlesSpeciesDownward.size()
    );
    unsigned int maxNumUpwardForProcs = max(
        host_sendParticlesSpeciesUpward.size(), 
        host_recvParticlesSpeciesUpward.size()
    );

    unsigned int maxNumDownward = 0, maxNumUpward = 0;
    MPI_Allreduce(&maxNumDownwardForProcs, &maxNumDownward, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&maxNumUpwardForProcs, &maxNumUpward, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
   

    thrust::host_vector<Particle> sendbufDownward(maxNumDownward);
    thrust::host_vector<Particle> sendbufUpward(maxNumUpward);
    thrust::host_vector<Particle> recvbufDownward(maxNumDownward);
    thrust::host_vector<Particle> recvbufUpward(maxNumUpward);

    for (unsigned int i = 0; i < host_sendParticlesSpeciesDownward.size(); i++) {
        sendbufDownward[i] = host_sendParticlesSpeciesDownward[i];
    }
    for (unsigned int i = 0; i < host_sendParticlesSpeciesUpward.size(); i++) {
        sendbufUpward[i] = host_sendParticlesSpeciesUpward[i];
    }


    MPI_Sendrecv(
        thrust::raw_pointer_cast(sendbufDownward.data()), 
        sendbufDownward.size(), 
        mPIInfo.mpi_particle_type, 
        down, 0, 
        thrust::raw_pointer_cast(recvbufDownward.data()), 
        recvbufDownward.size(), 
        mPIInfo.mpi_particle_type, 
        up, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        thrust::raw_pointer_cast(sendbufUpward.data()), 
        sendbufUpward.size(), 
        mPIInfo.mpi_particle_type, 
        up, 0, 
        thrust::raw_pointer_cast(recvbufUpward.data()),
        recvbufUpward.size(), 
        mPIInfo.mpi_particle_type, 
        down, 0, 
        MPI_COMM_WORLD, &st
    );


    for (unsigned int i = 0; i < host_recvParticlesSpeciesDownward.size(); i++) {
        host_recvParticlesSpeciesDownward[i] = recvbufDownward[i];
    }
    for (unsigned int i = 0; i < host_recvParticlesSpeciesUpward.size(); i++) {
        host_recvParticlesSpeciesUpward[i] = recvbufUpward[i];
    }
}

