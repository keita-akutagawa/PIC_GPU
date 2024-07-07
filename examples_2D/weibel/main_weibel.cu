#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "../../lib_pic2D_gpu_single/pic2D.hpp"


std::string directoryname = "results_weibel";
std::string filenameWithoutStep = "weibel";
std::ofstream logfile("results_weibel/log_weibel.txt");


const float c = 1.0f;
const float epsilon0 = 1.0f;
const float mu0 = 1.0f;

const int nx = 256;
const float dx = 1.0f;
const float xmin = 0.0f; 
const float xmax = nx * dx;

const int ny = 256;
const float dy = 1.0f;
const float ymin = 0.0f; 
const float ymax = ny * dy;

const float dt = 0.5f;

const int numberDensityIon = 20;
const int numberDensityElectron = 20;

const int totalNumIon = nx * ny * numberDensityIon;
const int totalNumElectron = nx * ny * numberDensityElectron;
const int totalNumParticles = totalNumIon + totalNumElectron;

const float B0 = 1.0f;

const float mRatio = 1.0f;
const float mElectron = 1.0f;
const float mIon = mRatio * mElectron;

const float tRatio = 100.0f;
const float tElectron = 0.5 * mElectron * pow(0.1f * c, 2);
const float tIon = 0.5 * mIon * pow(0.1f * c, 2);

const float qRatio = -1.0f;
const float qElectron = -1.0f * sqrt(epsilon0 * tElectron / static_cast<float>(numberDensityElectron));
const float qIon = qRatio * qElectron;

const float omegaPe = sqrt(static_cast<float>(numberDensityElectron) * pow(qElectron, 2) / mElectron / epsilon0);
const float omegaPi = sqrt(static_cast<float>(numberDensityIon) * pow(qIon, 2) / mIon / epsilon0);
const float omegaCe = abs(qElectron * B0 / mElectron);
const float omegaCi = qIon * B0 / mIon;

const float debyeLength = sqrt(epsilon0 * tElectron / static_cast<float>(numberDensityElectron) / pow(qElectron, 2));

const float vThIon = sqrt(2.0f * tIon / mIon);
const float vThElectron = sqrt(2.0f * tElectron / mElectron);
const float bulkVxIon = 0.0f;
const float bulkVyIon = 0.0f;
const float bulkVzIon = 0.0f;
const float bulkVxElectron = 0.0f * vThIon;
const float bulkVyElectron = 0.0f;
const float bulkVzElectron = 0.0f;

const int totalStep = 3000;
const int recordStep = 100;
float totalTime = 0.0f;


__constant__ float device_c;
__constant__ float device_epsilon0;
__constant__ float device_mu0;

__constant__ int device_nx;
__constant__ float device_dx;
__constant__ float device_xmin; 
__constant__ float device_xmax;

__constant__ int device_ny;
__constant__ float device_dy;
__constant__ float device_ymin; 
__constant__ float device_ymax;

__constant__ float device_dt;

__constant__ int device_numberDensityIon;
__constant__ int device_numberDensityElectron;

__constant__ int device_totalNumIon;
__constant__ int device_totalNumElectron;
__constant__ int device_totalNumParticles;

__constant__ float device_B0;

__constant__ float device_mRatio;
__constant__ float device_mIon;
__constant__ float device_mElectron;

__constant__ float device_tRatio;
__constant__ float device_tIon;
__constant__ float device_tElectron;

__constant__ float device_qRatio;
__constant__ float device_qIon;
__constant__ float device_qElectron;

__constant__ float device_omegaPe;
__constant__ float device_omegaPi;
__constant__ float device_omegaCe;
__constant__ float device_omegaCi;

__constant__ float device_debyeLength;

__constant__ float device_vThIon;
__constant__ float device_vThElectron;
__constant__ float device_bulkVxIon;
__constant__ float device_bulkVyIon;
__constant__ float device_bulkVzIon;
__constant__ float device_bulkVxElectron;
__constant__ float device_bulkVyElectron;
__constant__ float device_bulkVzElectron;

__constant__ int device_totalStep;
__device__ float device_totalTime;


__global__ void initializeField_kernel(
    ElectricField* E, MagneticField* B
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx && j < device_ny) {
        E[j + device_ny * i].eX = 0.0f;
        E[j + device_ny * i].eY = 0.0f;
        E[j + device_ny * i].eZ = 0.0f;
        B[j + device_ny * i].bX = 0.0f;
        B[j + device_ny * i].bY = 0.0f;
        B[j + device_ny * i].bZ = 0.0f;
    }
}

void PIC2D::initialize()
{
    initializeParticle.uniformForPositionX(
        0, totalNumIon, 0, particlesIon
    );
    initializeParticle.uniformForPositionX(
        0, totalNumElectron, 100, particlesElectron
    );
    initializeParticle.uniformForPositionY(
        0, totalNumIon, 200, particlesIon
    );
    initializeParticle.uniformForPositionY(
        0, totalNumElectron, 300, particlesElectron
    );

    initializeParticle.maxwellDistributionForVelocity(
        bulkVxIon, bulkVyIon, bulkVzIon, vThIon, vThIon, 5.0f * vThIon, 
        0, totalNumIon, 400, particlesIon
    );
    initializeParticle.maxwellDistributionForVelocity(
        bulkVxElectron, bulkVyElectron, bulkVzElectron, vThElectron, vThElectron, 5.0f * vThElectron, 
        0, totalNumElectron, 500, particlesElectron
    );


    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializeField_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), thrust::raw_pointer_cast(B.data())
    );

    cudaDeviceSynchronize();
}


int main()
{
    initializeDeviceConstants();

    std::cout << "total number of partices is " << totalNumParticles << std::endl;
    std::cout << std::setprecision(4) 
              << "omega_pe * t = " << totalStep * dt * omegaPe << std::endl;


    PIC2D pIC2D;

    pIC2D.initialize();

    for (int step = 0; step < totalStep+1; step++) {
        if (step % recordStep == 0) {
            std::cout << std::to_string(step) << " step done : total time is "
                      << std::setprecision(4) << totalTime
                      << std::endl;
            logfile << std::setprecision(6) << totalTime << std::endl;
            pIC2D.saveFields(
                directoryname, filenameWithoutStep, step
            );
            pIC2D.saveParticle(
                directoryname, filenameWithoutStep, step
            );
        }
        
        pIC2D.oneStep();

        totalTime += dt;
    }

    return 0;
}



