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
    offsets_particle[8]  = offsetof(Particle, isMPISendLeft);
    offsets_particle[9]  = offsetof(Particle, isMPISendRight);
    offsets_particle[10] = offsetof(Particle, isMPISendDown);
    offsets_particle[11] = offsetof(Particle, isMPISendUp);

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


void sendrecv_numParticle_x(
    const unsigned int& numForSendParticlesSpeciesLeft, 
    const unsigned int& numForSendParticlesSpeciesRight, 
    unsigned int& numForRecvParticlesSpeciesLeft, 
    unsigned int& numForRecvParticlesSpeciesRight, 
    MPIInfo& mPIInfo
)
{
    int left  = mPIInfo.getRank(-1, 0);
    int right = mPIInfo.getRank(1, 0);
    MPI_Status st;

    MPI_Sendrecv(
        &(numForSendParticlesSpeciesRight), 
        1, 
        MPI_UNSIGNED, 
        right, 0, 
        &(numForRecvParticlesSpeciesRight), 
        1, 
        MPI_UNSIGNED, 
        left, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        &(numForSendParticlesSpeciesLeft), 
        1, 
        MPI_UNSIGNED,  
        left, 0, 
        &(numForRecvParticlesSpeciesLeft), 
        1, 
        MPI_UNSIGNED, 
        right, 0, 
        MPI_COMM_WORLD, &st
    );
}


void sendrecv_particle_x(
    thrust::host_vector<Particle>& host_sendParticlesSpeciesLeft, 
    thrust::host_vector<Particle>& host_sendParticlesSpeciesRight, 
    thrust::host_vector<Particle>& host_recvParticlesSpeciesLeft, 
    thrust::host_vector<Particle>& host_recvParticlesSpeciesRight, 
    MPIInfo& mPIInfo
)
{
    int left  = mPIInfo.getRank(-1, 0);
    int right = mPIInfo.getRank(1, 0);
    MPI_Status st;


    unsigned int maxNumLeftForProcs = max(
        host_sendParticlesSpeciesLeft.size(), 
        host_recvParticlesSpeciesLeft.size()
    );
    unsigned int maxNumRightForProcs = max(
        host_sendParticlesSpeciesRight.size(), 
        host_recvParticlesSpeciesRight.size()
    );

    unsigned int maxNumLeft = 0, maxNumRight = 0;
    MPI_Allreduce(&maxNumLeftForProcs, &maxNumLeft, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&maxNumRightForProcs, &maxNumRight, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
    
    thrust::host_vector<Particle> sendbufLeft(maxNumLeft);
    thrust::host_vector<Particle> sendbufRight(maxNumRight);
    thrust::host_vector<Particle> recvbufLeft(maxNumLeft);
    thrust::host_vector<Particle> recvbufRight(maxNumRight);

    for (unsigned int i = 0; i < host_sendParticlesSpeciesLeft.size(); i++) {
        sendbufLeft[i] = host_sendParticlesSpeciesLeft[i];
    }
    for (unsigned int i = 0; i < host_sendParticlesSpeciesRight.size(); i++) {
        sendbufRight[i] = host_sendParticlesSpeciesRight[i];
    }
    

    MPI_Sendrecv(
        thrust::raw_pointer_cast(sendbufRight.data()), 
        sendbufRight.size(), 
        mPIInfo.mpi_particle_type, 
        right, 0, 
        thrust::raw_pointer_cast(recvbufRight.data()),
        recvbufRight.size(), 
        mPIInfo.mpi_particle_type, 
        left, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        thrust::raw_pointer_cast(sendbufLeft.data()), 
        sendbufLeft.size(), 
        mPIInfo.mpi_particle_type, 
        left, 0, 
        thrust::raw_pointer_cast(recvbufLeft.data()), 
        recvbufLeft.size(), 
        mPIInfo.mpi_particle_type, 
        right, 0, 
        MPI_COMM_WORLD, &st
    );


    for (unsigned int i = 0; i < host_recvParticlesSpeciesLeft.size(); i++) {
        host_recvParticlesSpeciesLeft[i] = recvbufLeft[i];
    }
    for (unsigned int i = 0; i < host_recvParticlesSpeciesRight.size(); i++) {
        host_recvParticlesSpeciesRight[i] = recvbufRight[i];
    }
}


void sendrecv_numParticle_corner(
    const unsigned int& numForSendParticlesSpeciesCornerLeftDown, 
    const unsigned int& numForSendParticlesSpeciesCornerRightDown, 
    const unsigned int& numForSendParticlesSpeciesCornerLeftUp, 
    const unsigned int& numForSendParticlesSpeciesCornerRightUp, 
    unsigned int& numForRecvParticlesSpeciesCornerLeftDown, 
    unsigned int& numForRecvParticlesSpeciesCornerRightDown, 
    unsigned int& numForRecvParticlesSpeciesCornerLeftUp, 
    unsigned int& numForRecvParticlesSpeciesCornerRightUp, 
    MPIInfo& mPIInfo
)
{
    int left  = mPIInfo.getRank(-1, 0);
    int right = mPIInfo.getRank(1, 0);
    MPI_Status st;

    MPI_Sendrecv(
        &(numForSendParticlesSpeciesCornerLeftDown), 
        1, 
        MPI_UNSIGNED,  
        left, 0, 
        &(numForRecvParticlesSpeciesCornerRightDown), 
        1, 
        MPI_UNSIGNED, 
        right, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        &(numForSendParticlesSpeciesCornerRightDown), 
        1, 
        MPI_UNSIGNED, 
        right, 0, 
        &(numForRecvParticlesSpeciesCornerLeftDown), 
        1, 
        MPI_UNSIGNED, 
        left, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        &(numForSendParticlesSpeciesCornerLeftUp), 
        1, 
        MPI_UNSIGNED,  
        left, 0, 
        &(numForRecvParticlesSpeciesCornerRightUp), 
        1, 
        MPI_UNSIGNED, 
        right, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        &(numForSendParticlesSpeciesCornerRightUp), 
        1, 
        MPI_UNSIGNED, 
        right, 0, 
        &(numForRecvParticlesSpeciesCornerLeftUp), 
        1, 
        MPI_UNSIGNED, 
        left, 0, 
        MPI_COMM_WORLD, &st
    );
}


void sendrecv_numParticle_y(
    const unsigned int& numForSendParticlesSpeciesDown, 
    const unsigned int& numForSendParticlesSpeciesUp, 
    unsigned int& numForRecvParticlesSpeciesDown, 
    unsigned int& numForRecvParticlesSpeciesUp, 
    MPIInfo& mPIInfo
)
{
    int down = mPIInfo.getRank(0, -1);
    int up   = mPIInfo.getRank(0, 1);
    MPI_Status st;

    MPI_Sendrecv(
        &(numForSendParticlesSpeciesUp), 
        1, 
        MPI_UNSIGNED, 
        up, 0, 
        &(numForRecvParticlesSpeciesUp), 
        1, 
        MPI_UNSIGNED, 
        down, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        &(numForSendParticlesSpeciesDown), 
        1, 
        MPI_UNSIGNED,  
        down, 0, 
        &(numForRecvParticlesSpeciesDown), 
        1, 
        MPI_UNSIGNED, 
        up, 0, 
        MPI_COMM_WORLD, &st
    );
}


void sendrecv_particle_y(
    thrust::host_vector<Particle>& host_sendParticlesSpeciesDown, 
    thrust::host_vector<Particle>& host_sendParticlesSpeciesUp, 
    thrust::host_vector<Particle>& host_recvParticlesSpeciesDown, 
    thrust::host_vector<Particle>& host_recvParticlesSpeciesUp, 
    MPIInfo& mPIInfo
)
{
    int down  = mPIInfo.getRank(0, -1);
    int up = mPIInfo.getRank(0, 1);
    MPI_Status st;


    unsigned int maxNumDownForProcs = max(
        host_sendParticlesSpeciesDown.size(), 
        host_recvParticlesSpeciesDown.size()
    );
    unsigned int maxNumUpForProcs = max(
        host_sendParticlesSpeciesUp.size(), 
        host_recvParticlesSpeciesUp.size()
    );

    unsigned int maxNumDown = 0, maxNumUp = 0;
    MPI_Allreduce(&maxNumDownForProcs, &maxNumDown, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&maxNumUpForProcs, &maxNumUp, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
   

    thrust::host_vector<Particle> sendbufDown(maxNumDown);
    thrust::host_vector<Particle> sendbufUp(maxNumUp);
    thrust::host_vector<Particle> recvbufDown(maxNumDown);
    thrust::host_vector<Particle> recvbufUp(maxNumUp);

    for (unsigned int i = 0; i < host_sendParticlesSpeciesDown.size(); i++) {
        sendbufDown[i] = host_sendParticlesSpeciesDown[i];
    }
    for (unsigned int i = 0; i < host_sendParticlesSpeciesUp.size(); i++) {
        sendbufUp[i] = host_sendParticlesSpeciesUp[i];
    }


    MPI_Sendrecv(
        thrust::raw_pointer_cast(sendbufDown.data()), 
        sendbufDown.size(), 
        mPIInfo.mpi_particle_type, 
        down, 0, 
        thrust::raw_pointer_cast(recvbufDown.data()), 
        recvbufDown.size(), 
        mPIInfo.mpi_particle_type, 
        up, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        thrust::raw_pointer_cast(sendbufUp.data()), 
        sendbufUp.size(), 
        mPIInfo.mpi_particle_type, 
        up, 0, 
        thrust::raw_pointer_cast(recvbufUp.data()),
        recvbufUp.size(), 
        mPIInfo.mpi_particle_type, 
        down, 0, 
        MPI_COMM_WORLD, &st
    );


    for (unsigned int i = 0; i < host_recvParticlesSpeciesDown.size(); i++) {
        host_recvParticlesSpeciesDown[i] = recvbufDown[i];
    }
    for (unsigned int i = 0; i < host_recvParticlesSpeciesUp.size(); i++) {
        host_recvParticlesSpeciesUp[i] = recvbufUp[i];
    }
}

