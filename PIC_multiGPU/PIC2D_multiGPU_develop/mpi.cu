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


void setupInfo(MPIInfo& mPIInfo, int buffer, int mpiBufNumParticles)
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
    mPIInfo.localGridX = rank / mPIInfo.gridY;
    mPIInfo.localGridY = rank % mPIInfo.gridY;
    mPIInfo.localNx = nx / mPIInfo.gridX;
    mPIInfo.localNy = ny / mPIInfo.gridY;
    mPIInfo.buffer = buffer;
    mPIInfo.localSizeX = mPIInfo.localNx + 2 * mPIInfo.buffer;
    mPIInfo.localSizeY = mPIInfo.localNy + 2 * mPIInfo.buffer;
    mPIInfo.mpiBufNumParticles = mpiBufNumParticles; 


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

void sendrecv_magneticField_x(
    thrust::device_vector<MagneticField>& B, 
    thrust::device_vector<MagneticField>& sendMagneticFieldLeft, 
    thrust::device_vector<MagneticField>& sendMagneticFieldRight, 
    thrust::device_vector<MagneticField>& recvMagneticFieldLeft, 
    thrust::device_vector<MagneticField>& recvMagneticFieldRight, 
    MPIInfo& mPIInfo
)
{
    sendrecv_field_x(
        B, 
        sendMagneticFieldLeft, sendMagneticFieldRight, 
        recvMagneticFieldLeft, recvMagneticFieldRight, 
        mPIInfo
    );
}

void sendrecv_magneticField_y(
    thrust::device_vector<MagneticField>& B, 
    thrust::device_vector<MagneticField>& sendMagneticFieldDown, 
    thrust::device_vector<MagneticField>& sendMagneticFieldUp, 
    thrust::device_vector<MagneticField>& recvMagneticFieldDown, 
    thrust::device_vector<MagneticField>& recvMagneticFieldUp, 
    MPIInfo& mPIInfo
)
{
    sendrecv_field_y(
        B, 
        sendMagneticFieldDown, sendMagneticFieldUp, 
        recvMagneticFieldDown, recvMagneticFieldUp, 
        mPIInfo
    );
}


void sendrecv_electricField_x(
    thrust::device_vector<ElectricField>& E, 
    thrust::device_vector<ElectricField>& sendElectricFieldLeft, 
    thrust::device_vector<ElectricField>& sendElectricFieldRight, 
    thrust::device_vector<ElectricField>& recvElectricFieldLeft, 
    thrust::device_vector<ElectricField>& recvElectricFieldRight,
    MPIInfo& mPIInfo
)
{
    sendrecv_field_x(
        E, 
        sendElectricFieldLeft, sendElectricFieldRight, 
        recvElectricFieldLeft, recvElectricFieldRight, 
        mPIInfo
    );
}

void sendrecv_electricField_y(
    thrust::device_vector<ElectricField>& E, 
    thrust::device_vector<ElectricField>& sendElectricFieldDown, 
    thrust::device_vector<ElectricField>& sendElectricFieldUp, 
    thrust::device_vector<ElectricField>& recvElectricFieldDown, 
    thrust::device_vector<ElectricField>& recvElectricFieldUp, 
    MPIInfo& mPIInfo
)
{
    sendrecv_field_y(
        E, 
        sendElectricFieldDown, sendElectricFieldUp, 
        recvElectricFieldDown, recvElectricFieldUp, 
        mPIInfo
    );
}


void sendrecv_currentField_x(
    thrust::device_vector<CurrentField>& current,  
    thrust::device_vector<CurrentField>& sendCurrentFieldLeft, 
    thrust::device_vector<CurrentField>& sendCurrentFieldRight, 
    thrust::device_vector<CurrentField>& recvCurrentFieldLeft, 
    thrust::device_vector<CurrentField>& recvCurrentFieldRight, 
    MPIInfo& mPIInfo
)
{
    sendrecv_field_x(
        current, 
        sendCurrentFieldLeft, sendCurrentFieldRight, 
        recvCurrentFieldLeft, recvCurrentFieldRight, 
        mPIInfo
    );
}

void sendrecv_currentField_y(
    thrust::device_vector<CurrentField>& current,  
    thrust::device_vector<CurrentField>& sendCurrentFieldDown, 
    thrust::device_vector<CurrentField>& sendCurrentFieldUp, 
    thrust::device_vector<CurrentField>& recvCurrentFieldDown, 
    thrust::device_vector<CurrentField>& recvCurrentFieldUp, 
    MPIInfo& mPIInfo
)
{
    sendrecv_field_y(
        current, 
        sendCurrentFieldDown, sendCurrentFieldUp, 
        recvCurrentFieldDown, recvCurrentFieldUp, 
        mPIInfo
    );
}



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
        &(numForSendParticlesSpeciesLeft), 
        1, 
        MPI_UNSIGNED,  
        left, 0, 
        &(numForRecvParticlesSpeciesRight), 
        1, 
        MPI_UNSIGNED, 
        right, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        &(numForSendParticlesSpeciesRight), 
        1, 
        MPI_UNSIGNED, 
        right, 0, 
        &(numForRecvParticlesSpeciesLeft), 
        1, 
        MPI_UNSIGNED, 
        left, 0, 
        MPI_COMM_WORLD, &st
    );
}


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
)
{
    int left  = mPIInfo.getRank(-1, 0);
    int right = mPIInfo.getRank(1, 0);
    MPI_Status st;


    unsigned int maxNumSendLeftRecvRightForProcs = max(
        numForSendParticlesSpeciesLeft, 
        numForRecvParticlesSpeciesRight 
    );
    unsigned int maxNumSendRightRecvLeftForProcs = max(
        numForSendParticlesSpeciesRight, 
        numForRecvParticlesSpeciesLeft
    );

    unsigned int maxNumSendLeftRecvRight = 0, maxNumSendRightRecvLeft = 0;
    MPI_Allreduce(&maxNumSendLeftRecvRightForProcs, &maxNumSendLeftRecvRight, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&maxNumSendRightRecvLeftForProcs, &maxNumSendRightRecvLeft, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);

    MPI_Sendrecv(
        thrust::raw_pointer_cast(sendParticlesSpeciesLeft.data()), 
        numForSendParticlesSpeciesLeft, 
        mPIInfo.mpi_particle_type, 
        left, 0, 
        thrust::raw_pointer_cast(recvParticlesSpeciesRight.data()), 
        numForRecvParticlesSpeciesRight, 
        mPIInfo.mpi_particle_type, 
        right, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        thrust::raw_pointer_cast(sendParticlesSpeciesRight.data()), 
        numForSendParticlesSpeciesRight, 
        mPIInfo.mpi_particle_type, 
        right, 0, 
        thrust::raw_pointer_cast(recvParticlesSpeciesLeft.data()),
        numForRecvParticlesSpeciesLeft, 
        mPIInfo.mpi_particle_type, 
        left, 0, 
        MPI_COMM_WORLD, &st
    );
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
        &(numForSendParticlesSpeciesDown), 
        1, 
        MPI_UNSIGNED,  
        down, 0, 
        &(numForRecvParticlesSpeciesUp), 
        1, 
        MPI_UNSIGNED, 
        up, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        &(numForSendParticlesSpeciesUp), 
        1, 
        MPI_UNSIGNED, 
        up, 0, 
        &(numForRecvParticlesSpeciesDown), 
        1, 
        MPI_UNSIGNED, 
        down, 0, 
        MPI_COMM_WORLD, &st
    );
}


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
)
{
    int down  = mPIInfo.getRank(0, -1);
    int up = mPIInfo.getRank(0, 1);
    MPI_Status st;


    unsigned int maxNumSendDownRecvUpForProcs = max(
        numForSendParticlesSpeciesDown, 
        numForRecvParticlesSpeciesUp 
    );
    unsigned int maxNumSendUpRecvDownForProcs = max(
        numForSendParticlesSpeciesUp, 
        numForRecvParticlesSpeciesDown
    );

    unsigned int maxNumSendDownRecvUp = 0, maxNumSendUpRecvDown = 0;
    MPI_Allreduce(&maxNumSendDownRecvUpForProcs, &maxNumSendDownRecvUp, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&maxNumSendUpRecvDownForProcs, &maxNumSendUpRecvDown, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);

    MPI_Sendrecv(
        thrust::raw_pointer_cast(sendParticlesSpeciesDown.data()), 
        numForSendParticlesSpeciesDown, 
        mPIInfo.mpi_particle_type, 
        down, 0, 
        thrust::raw_pointer_cast(recvParticlesSpeciesUp.data()), 
        numForRecvParticlesSpeciesUp, 
        mPIInfo.mpi_particle_type, 
        up, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        thrust::raw_pointer_cast(sendParticlesSpeciesUp.data()), 
        numForSendParticlesSpeciesUp, 
        mPIInfo.mpi_particle_type, 
        up, 0, 
        thrust::raw_pointer_cast(recvParticlesSpeciesDown.data()),
        numForRecvParticlesSpeciesDown, 
        mPIInfo.mpi_particle_type, 
        down, 0, 
        MPI_COMM_WORLD, &st
    );
}

