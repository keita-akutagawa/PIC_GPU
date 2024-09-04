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
}


//////////////////////////////////////////////////

// sendrecv(field用)はヘッダーファイルにある。
// templeteを使用したため


void sendrecv_particle(
    thrust::device_vector<Particle>& particlesSpecies, 
    thrust::device_vector<Particle>& sendrecvParticlesSpeciesLeftToRight, 
    thrust::device_vector<Particle>& sendrecvParticlesSpeciesRightToLeft, 
    MPIInfo& mPIInfo)
{
    int left = mPIInfo.getRank(-1);
    int right = mPIInfo.getRank(1);
    MPI_Status st;


    

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


