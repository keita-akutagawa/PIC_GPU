#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "../../lib_pic2D_gpu_single_triple/pic2D.hpp"
#include <cuda_runtime.h>


std::string directoryname = "results_mr_triple_4-16-20";
std::string filenameWithoutStep = "mr";
std::ofstream logfile("results_mr_triple_4-16-20/log_mr.txt");

const int totalStep = 60000;
const int fieldRecordStep = 400;
const bool isParticleRecord = false;
const int particleRecordStep = totalStep;
float totalTime = 0.0f;

const float c = 1.0f;
const float epsilon0 = 1.0f;
const float mu0 = 1.0f;
const float dOfLangdonMarderTypeCorrection = 0.005f;

const int numberDensityIon = 50;
const int numberDensityHeavyIon = 2; //backgroundでの数密度を設定すること
const int numberDensityElectron = 50;

const float B0 = sqrt(static_cast<float>(numberDensityElectron)) / 1.0f;

const float mRatio = 1.0f;
const float mElectron = 1.0f;
const float mIon = mRatio * mElectron;
const float mHeavyIon = mIon * 100.0f;

const float tRatio = 1.0f;
const float tElectron = (B0 * B0 / 2.0 / mu0) / (numberDensityIon + numberDensityElectron * tRatio);
const float tIon = tRatio * tElectron;
const float tHeavyIon = tIon;

const float qRatio = -1.0f;
const float qElectron = -1.0f * sqrt(epsilon0 * tElectron / static_cast<float>(numberDensityElectron));
const float qIon = qRatio * qElectron;
const float qHeavyIon = qIon;

const float omegaPe = sqrt(static_cast<float>(numberDensityElectron) * pow(qElectron, 2) / mElectron / epsilon0);
const float omegaPi = sqrt(static_cast<float>(numberDensityIon) * pow(qIon, 2) / mIon / epsilon0);
const float omegaCe = abs(qElectron * B0 / mElectron);
const float omegaCi = qIon * B0 / mIon;

const float debyeLength = sqrt(epsilon0 * tElectron / static_cast<float>(numberDensityElectron) / pow(qElectron, 2));
//追加
const float ionInertialLength = c / omegaPi;

const int nx = int(1000.0f * ionInertialLength);
const float dx = 1.0f;
const float xmin = 0.0f * dx; 
const float xmax = nx * dx - 0.0f * dx;
//const float xmin = 0.5f * dx; 
//const float xmax = nx * dx - 1.5f * dx;

const int ny = int(500.0f * ionInertialLength);
const float dy = 1.0f;
const float ymin = 1.0f * dy; 
const float ymax = ny * dy - 1.5f * dy;

const float dt = 0.25f;

//追加
const float sheatThickness = 2.0f * ionInertialLength;
const float triggerRatio = 0.0f;
const float xPointPosition = 250.0f * ionInertialLength;

//追加
const int harrisNumIon = int(nx * numberDensityIon * 2.0f * sheatThickness);
const int backgroundNumIon = int(0.16f * nx * ny * numberDensityIon);
const long long totalNumIon = harrisNumIon + backgroundNumIon;
const int harrisNumElectron = int(nx * numberDensityElectron * 2.0f * sheatThickness);
const int backgroundNumElectron = int(0.2f * nx * ny * numberDensityElectron);
const long long totalNumElectron = harrisNumElectron + backgroundNumElectron;
const int backgroundNumHeavyIon = int(nx * ny * numberDensityHeavyIon);
const long long totalNumHeavyIon = backgroundNumHeavyIon;
const long long totalNumParticles = totalNumIon + totalNumElectron + totalNumHeavyIon;

const float vThIon = sqrt(2.0f * tIon / mIon);
const float vThElectron = sqrt(2.0f * tElectron / mElectron);
const float vThHeavyIon = sqrt(2.0f * tHeavyIon / mHeavyIon);

const float bulkVxElectron = 0.0f;
const float bulkVyElectron = 0.0f;
const float bulkVzElectron = c * debyeLength / sheatThickness * sqrt(2.0f / (1.0f + 1.0f/tRatio));
const float bulkVxIon = -bulkVxElectron / tRatio;
const float bulkVyIon = -bulkVyElectron / tRatio;
const float bulkVzIon = -bulkVzElectron / tRatio;

const float vThIonB = sqrt(2.0f * tIon * 0.2f / mIon);
const float vThElectronB = sqrt(2.0f * tElectron * 0.2f / mElectron);
const float vThHeavyIonB = sqrt(2.0f * tHeavyIon * 0.2f / mHeavyIon);

const float bulkVxElectronB = 0.0f;
const float bulkVyElectronB = 0.0f;
const float bulkVzElectronB = 0.0f;
const float bulkVxIonB = 0.0f;
const float bulkVyIonB = 0.0f;
const float bulkVzIonB = 0.0f;
const float bulkVxHeavyIonB = 0.0f;
const float bulkVyHeavyIonB = 0.0f;
const float bulkVzHeavyIonB = 0.0f;


__constant__ int device_totalStep;
__constant__ int device_recordStep;
__device__ float device_totalTime;

__constant__ float device_c;
__constant__ float device_epsilon0;
__constant__ float device_mu0;
__constant__ float device_dOfLangdonMarderTypeCorrection;

__constant__ int device_numberDensityIon;
__constant__ int device_numberDensityElectron;
__constant__ int device_numberDensityHeavyIon;

__constant__ float device_B0;

__constant__ float device_mRatio;
__constant__ float device_mElectron;
__constant__ float device_mIon;
__constant__ float device_mHeavyIon;

__constant__ float device_tRatio;
__constant__ float device_tElectron;
__constant__ float device_tIon;
__constant__ float device_tHeavyIon;

__constant__ float device_qRatio;
__constant__ float device_qElectron;
__constant__ float device_qIon;
__constant__ float device_qHeavyIon;

__constant__ float device_omegaPe;
__constant__ float device_omegaPi;
__constant__ float device_omegaCe;
__constant__ float device_omegaCi;

__constant__ float device_debyeLength;
//追加
__constant__ float device_ionInertialLength;

__constant__ int device_nx;
__constant__ float device_dx;
__constant__ float device_xmin;
__constant__ float device_xmax;

__constant__ int device_ny;
__constant__ float device_dy;
__constant__ float device_ymin;
__constant__ float device_ymax;

__constant__ float device_dt;

//追加
__constant__ float device_sheatThickness;
__constant__ float device_triggerRatio;
__constant__ float device_xPointPosition;

//追加
__constant__ int device_harrisNumIon;
__constant__ int device_backgroundNumIon;
__constant__ long long device_totalNumIon;
__constant__ int device_harrisNumElectron;
__constant__ int device_backgroundNumElectron;
__constant__ long long device_totalNumElectron;
__constant__ int device_backgroundNumHeavyIon;
__constant__ long long device_totalNumHeavyIon;
__constant__ long long device_totalNumParticles;

__constant__ float device_vThIon;
__constant__ float device_vThElectron;
__constant__ float device_vThHeavyIon;
__constant__ float device_bulkVxElectron;
__constant__ float device_bulkVyElectron;
__constant__ float device_bulkVzElectron;
__constant__ float device_bulkVxIon;
__constant__ float device_bulkVyIon;
__constant__ float device_bulkVzIon;
__constant__ float device_bulkVxHeavyIon;
__constant__ float device_bulkVyHeavyIon;
__constant__ float device_bulkVzHeavyIon;

__constant__ float device_vThIonB;
__constant__ float device_vThElectronB;
__constant__ float device_vThHeavyIonB;

__constant__ float device_bulkVxElectronB;
__constant__ float device_bulkVyElectronB;
__constant__ float device_bulkVzElectronB;
__constant__ float device_bulkVxIonB;
__constant__ float device_bulkVyIonB;
__constant__ float device_bulkVzIonB;
__constant__ float device_bulkVxHeavyIonB;
__constant__ float device_bulkVyHeavyIonB;
__constant__ float device_bulkVzHeavyIonB;


__global__ void initializeField_kernel(
    ElectricField* E, MagneticField* B
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx && j < device_ny) {
        float yCenter = 0.5f * (device_ymax - device_ymin) + device_ymin;
        E[j + device_ny * i].eX = 0.0f;
        E[j + device_ny * i].eY = 0.0f;
        E[j + device_ny * i].eZ = 0.0f;
        B[j + device_ny * i].bX = device_B0 * tanh((j * device_dy - yCenter) / device_sheatThickness)
                                - device_B0 * device_triggerRatio * (j * device_dy - yCenter) / device_sheatThickness
                                * exp(-(pow((i * device_dx - device_xPointPosition), 2) + pow((j * device_dy - yCenter), 2))
                                / pow(2.0f * device_sheatThickness, 2));
        B[j + device_ny * i].bY = device_B0 * device_triggerRatio * (i * device_dx - device_xPointPosition) / device_sheatThickness
                                * exp(-(pow((i * device_dx - device_xPointPosition), 2) + pow((j * device_dy - yCenter), 2))
                                / pow(2.0f * device_sheatThickness, 2)); 
        B[j + device_ny * i].bZ = 0.0f * device_B0;
    }
}

void PIC2D::initialize()
{
    cudaMemcpyToSymbol(device_ionInertialLength, &ionInertialLength, sizeof(float));
    cudaMemcpyToSymbol(device_sheatThickness, &sheatThickness, sizeof(float));
    cudaMemcpyToSymbol(device_triggerRatio, &triggerRatio, sizeof(float));
    cudaMemcpyToSymbol(device_xPointPosition, &xPointPosition, sizeof(float));
    cudaMemcpyToSymbol(device_harrisNumIon, &harrisNumIon, sizeof(int));
    cudaMemcpyToSymbol(device_backgroundNumIon, &backgroundNumIon, sizeof(int));
    cudaMemcpyToSymbol(device_harrisNumElectron, &harrisNumElectron, sizeof(int));
    cudaMemcpyToSymbol(device_backgroundNumElectron, &backgroundNumElectron, sizeof(int));
    cudaMemcpyToSymbol(device_backgroundNumHeavyIon, &backgroundNumHeavyIon, sizeof(int));

    initializeParticle.uniformForPositionX(
        0, totalNumIon, 0, particlesIon
    );
    initializeParticle.uniformForPositionX(
        0, totalNumElectron, 100, particlesElectron
    );
    initializeParticle.uniformForPositionX(
        0, totalNumHeavyIon, 200, particlesHeavyIon
    );

    initializeParticle.harrisForPositionY(
        0, harrisNumIon, 300, sheatThickness, particlesIon
    );
    initializeParticle.uniformForPositionY(
        harrisNumIon, totalNumIon, 400,particlesIon
    );
    initializeParticle.harrisForPositionY(
        0, harrisNumElectron, 500, sheatThickness, particlesElectron
    );
    initializeParticle.uniformForPositionY(
        harrisNumElectron, totalNumElectron, 600, particlesElectron
    );
    initializeParticle.uniformForPositionY(
        0, totalNumHeavyIon, 700, particlesHeavyIon
    );

    initializeParticle.maxwellDistributionForVelocity(
        bulkVxIon, bulkVyIon, bulkVzIon, vThIon, vThIon, vThIon, 
        0, harrisNumIon, 800, particlesIon
    );
    initializeParticle.maxwellDistributionForVelocity(
        bulkVxIonB, bulkVyIonB, bulkVzIonB, vThIonB, vThIonB, vThIonB, 
        harrisNumIon, totalNumIon, 900, particlesIon
    );
    initializeParticle.maxwellDistributionForVelocity(
        bulkVxElectron, bulkVyElectron, bulkVzElectron, vThElectron, vThElectron, vThElectron, 
        0, harrisNumElectron, 1000, particlesElectron
    );
    initializeParticle.maxwellDistributionForVelocity(
        bulkVxElectronB, bulkVyElectronB, bulkVzElectronB, vThElectronB, vThElectronB, vThElectronB, 
        harrisNumElectron, totalNumElectron, 1100, particlesElectron
    );
    initializeParticle.maxwellDistributionForVelocity(
        bulkVxHeavyIonB, bulkVyHeavyIonB, bulkVzHeavyIonB, vThHeavyIonB, vThHeavyIonB, vThHeavyIonB, 
        0, totalNumHeavyIon, 1200, particlesHeavyIon
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
              << "Omega_ci * t = " << totalStep * dt * omegaCi << std::endl;


    PIC2D pIC2D;

    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaError_t status = cudaMemGetInfo(&free_mem, &total_mem);

    std::cout << "Free memory: " << free_mem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Total memory: " << total_mem / (1024 * 1024) << " MB" << std::endl;


    pIC2D.initialize();

    for (int step = 0; step < totalStep+1; step++) {
        if (step % fieldRecordStep == 0) {
            std::cout << std::to_string(step) << " step done : total time is "
                      << std::setprecision(4) << step * dt * omegaCi
                      << " [Omega_ci * t]"
                      << std::endl;
            logfile << std::setprecision(6) << totalTime << std::endl;
            pIC2D.saveFields(
                directoryname, filenameWithoutStep, step
            );
            pIC2D.saveMoments(
                directoryname, filenameWithoutStep, step
            );
        }
        if (isParticleRecord && step % particleRecordStep == 0) {
            pIC2D.saveParticle(
                directoryname, filenameWithoutStep, step
            );
        }
        
        pIC2D.oneStepPeriodicXWallY();

        if (step % 100 == 0) {
            pIC2D.sortParticle();
        }

        totalTime += dt;
    }

    return 0;
}



