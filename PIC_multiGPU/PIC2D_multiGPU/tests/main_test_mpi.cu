#include "../particle_struct.hpp"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <mpi.h>


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0, procs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    cudaSetDevice(rank);

    MPI_Datatype mpi_particle_type;

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
    MPI_Type_create_struct(12, block_lengths_particle, offsets_particle, types_particle, &mpi_particle_type);
    MPI_Type_commit(&mpi_particle_type);


    thrust::host_vector<Particle> host_send(10000), host_recv(10000);
    thrust::device_vector<Particle> device_send(10000), device_recv(10000);

    if (rank == 0) {
        host_send[77].x = 100.0f;
        host_send[77].y = 200.0f;
        host_send[77].z = 300.0f;
        host_send[77].vx = 400.0f;
        host_send[77].vy = 500.0f;
        host_send[77].vz = 600.0f;
        host_send[77].isMPISendLeftToRight = true;
        host_send[77].isMPISendDownToUp = true;
    }
    if (rank == 1) {
        host_send[777].x = 100.0f;
        host_send[777].y = 200.0f;
        host_send[777].z = 300.0f;
        host_send[777].vx = 400.0f;
        host_send[777].vy = 500.0f;
        host_send[777].vz = 600.0f;
        host_send[777].isMPISendLeftToRight = true;
        host_send[777].isMPISendDownToUp = true;
    }
    device_send = host_send;

    int right = (rank + 1) % procs;
    int left = (rank - 1 + procs) % procs;
    MPI_Status st;

    MPI_Sendrecv(
        thrust::raw_pointer_cast(device_send.data()), 
        1000, 
        mpi_particle_type, 
        right, 0, 
        thrust::raw_pointer_cast(device_recv.data()),
        1000, 
        mpi_particle_type, 
        left, 0, 
        MPI_COMM_WORLD, &st
    );
    MPI_Sendrecv(
        thrust::raw_pointer_cast(device_send.data()), 
        1000, 
        mpi_particle_type, 
        left, 0, 
        thrust::raw_pointer_cast(device_recv.data()),
        1000, 
        mpi_particle_type, 
        right, 0, 
        MPI_COMM_WORLD, &st
    );

    host_recv = device_recv;

    if (rank == 1) {
        std::cout << host_recv[77].x << std::endl;
        std::cout << host_recv[77].y << std::endl;
        std::cout << host_recv[77].z << std::endl;
        std::cout << host_recv[77].vx << std::endl;
        std::cout << host_recv[77].vy << std::endl;
        std::cout << host_recv[77].vz << std::endl;
        std::cout << host_recv[77].isMPISendRightToLeft << std::endl;
        std::cout << host_recv[77].isMPISendLeftToRight << std::endl;
        std::cout << host_recv[77].isMPISendDownToUp << std::endl;
        std::cout << host_recv[77].isMPISendUpToDown << std::endl;
    }
    if (rank == 0) {
        std::cout << host_recv[777].x << std::endl;
        std::cout << host_recv[777].y << std::endl;
        std::cout << host_recv[777].z << std::endl;
        std::cout << host_recv[777].vx << std::endl;
        std::cout << host_recv[777].vy << std::endl;
        std::cout << host_recv[777].vz << std::endl;
        std::cout << host_recv[777].isMPISendRightToLeft << std::endl;
        std::cout << host_recv[777].isMPISendLeftToRight << std::endl;
        std::cout << host_recv[777].isMPISendDownToUp << std::endl;
        std::cout << host_recv[777].isMPISendUpToDown << std::endl;
    }

    MPI_Finalize();

    if (rank == 0) {
        std::cout << "program was completed!" << std::endl;
    }

    return 0;
}


