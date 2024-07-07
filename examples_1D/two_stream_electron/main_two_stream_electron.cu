#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "../../lib_pic1D_gpu_single/pic1D.hpp"


std::string directoryname = "results_two_stream_electron";
std::string filenameWithoutStep = "two_stream_electron";
std::ofstream logfile("results_two_stream_electron/log_two_stream_electron.txt");


const double c = 1.0;
const double epsilon0 = 1.0;
const double mu0 = 1.0;

const int nx = 512;
const double dx = 1.0;
const double xmin = 0.0; 
const double xmax = nx * dx;

const double dt = 0.5;

const int numberDensityIon = 100;
const int numberDensityElectron = 100;

const int totalNumIon = nx * numberDensityIon;
//追加
const int totalNumElectronBeam1 = nx * numberDensityElectron / 2;
const int totalNumElectronBeam2 = nx * numberDensityElectron / 2;
const int totalNumElectron = totalNumElectronBeam1 + totalNumElectronBeam2;
const int totalNumParticles = totalNumIon + totalNumElectron;

const double B0 = sqrt(static_cast<double>(numberDensityElectron)) / 10.0;

const double mRatio = 100.0;
const double mElectron = 1.0;
const double mIon = mRatio * mElectron;

const double tRatio = 100.0;
const double tElectron = 0.5 * mElectron * pow(0.01 * c, 2);
const double tIon = tRatio * tElectron;

const double qRatio = -1.0;
const double qElectron = -1.0 * sqrt(epsilon0 * tElectron / static_cast<double>(numberDensityElectron));
const double qIon = qRatio * qElectron;

const double omegaPe = sqrt(static_cast<double>(numberDensityElectron) * pow(qElectron, 2) / mElectron / epsilon0);
const double omegaPi = sqrt(static_cast<double>(numberDensityIon) * pow(qIon, 2) / mIon / epsilon0);
const double omegaCe = abs(qElectron * B0 / mElectron);
const double omegaCi = qIon * B0 / mIon;

const double debyeLength = sqrt(epsilon0 * tElectron / static_cast<double>(numberDensityElectron) / pow(qElectron, 2));

const double vThIon = sqrt(2.0 * tIon / mIon);
const double vThElectron = sqrt(2.0 * tElectron / mElectron);
const double bulkVxIon = 0.0;
const double bulkVyIon = 0.0;
const double bulkVzIon = 0.0;
const double bulkVxElectron = -10.0 * vThIon;
const double bulkVyElectron = 0.0;
const double bulkVzElectron = 0.0;
//追加
const double bulkVxElectronBeam = 10.0 * vThIon;
const double bulkVyElectronBeam = 0.0;
const double bulkVzElectronBeam = 0.0;

const int totalStep = 10000;
const int recordStep = 100;
double totalTime = 0.0;


__constant__ double device_c;
__constant__ double device_epsilon0;
__constant__ double device_mu0;

__constant__ int device_nx;
__constant__ double device_dx;
__constant__ double device_xmin; 
__constant__ double device_xmax;

__constant__ double device_dt;

__constant__ int device_numberDensityIon;
__constant__ int device_numberDensityElectron;

__constant__ int device_totalNumIon;
//追加
__constant__ int device_totalNumElectronBeam1;
__constant__ int device_totalNumElectronBeam2;
__constant__ int device_totalNumElectron;
__constant__ int device_totalNumParticles;

__constant__ double device_B0;

__constant__ double device_mRatio;
__constant__ double device_mIon;
__constant__ double device_mElectron;

__constant__ double device_tRatio;
__constant__ double device_tIon;
__constant__ double device_tElectron;

__constant__ double device_qRatio;
__constant__ double device_qIon;
__constant__ double device_qElectron;

__constant__ double device_omegaPe;
__constant__ double device_omegaPi;
__constant__ double device_omegaCe;
__constant__ double device_omegaCi;

__constant__ double device_debyeLength;

__constant__ double device_vThIon;
__constant__ double device_vThElectron;
__constant__ double device_bulkVxIon;
__constant__ double device_bulkVyIon;
__constant__ double device_bulkVzIon;
__constant__ double device_bulkVxElectron;
__constant__ double device_bulkVyElectron;
__constant__ double device_bulkVzElectron;
//追加
__constant__ double device_bulkVxElectronBeam;
__constant__ double device_bulkVyElectronBeam;
__constant__ double device_bulkVzElectronBeam;

__constant__ int device_totalStep;
__device__ double device_totalTime;


__global__ void initializeField_kernel(
    ElectricField* E, MagneticField* B
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < device_nx) {
        E[i].eX = 0.0;
        E[i].eY = 0.0;
        E[i].eZ = 0.0;
        B[i].bX = device_B0;
        B[i].bY = 0.0;
        B[i].bZ = 0.0;
    }
}

void PIC1D::initialize()
{
    cudaMemcpyToSymbol(device_totalNumElectronBeam1, &totalNumElectronBeam1, sizeof(int));
    cudaMemcpyToSymbol(device_totalNumElectronBeam2, &totalNumElectronBeam2, sizeof(int));
    cudaMemcpyToSymbol(device_bulkVxElectronBeam, &bulkVxElectronBeam, sizeof(double));
    cudaMemcpyToSymbol(device_bulkVxElectronBeam, &bulkVxElectronBeam, sizeof(double));
    cudaMemcpyToSymbol(device_bulkVxElectronBeam, &bulkVxElectronBeam, sizeof(double));


    initializeParticle.uniformForPositionX(
        0, totalNumIon, 0, particlesIon
    );
    initializeParticle.uniformForPositionX(
        0, totalNumElectron, 10000, particlesElectron
    );

    initializeParticle.maxwellDistributionForVelocity(
        bulkVxIon, bulkVyIon, bulkVzIon, vThIon, 
        0, totalNumIon, 20000, particlesIon
    );
    initializeParticle.maxwellDistributionForVelocity(
        bulkVxElectron, bulkVyElectron, bulkVzElectron, vThElectron, 
        0, totalNumElectronBeam1, 30000, particlesElectron
    );
    initializeParticle.maxwellDistributionForVelocity(
        bulkVxElectronBeam, bulkVyElectronBeam, bulkVzElectronBeam, vThElectron, 
        totalNumElectronBeam1, totalNumElectron, 40000, particlesElectron
    );


    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x);

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


    PIC1D pIC1D;

    pIC1D.initialize();

    for (int step = 0; step < totalStep+1; step++) {
        if (step % recordStep == 0) {
            std::cout << std::to_string(step) << " step done : total time is "
                      << std::setprecision(4) << totalTime
                      << std::endl;
            logfile << std::setprecision(6) << totalTime << std::endl;
            pIC1D.saveFields(
                directoryname, filenameWithoutStep, step
            );
            pIC1D.saveParticle(
                directoryname, filenameWithoutStep, step
            );
        }

        pIC1D.oneStep();

        totalTime += dt;
    }

    return 0;
}



