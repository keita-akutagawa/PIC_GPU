#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "../../lib_pic2D_gpu_single/pic2D.hpp"
#include <cuda_runtime.h>


std::string directoryname = "/cfca-work/akutagawakt/PIC/results_mr_mr8_fadeev_harris";
std::string filenameWithoutStep = "mr";
std::ofstream logfile("/cfca-work/akutagawakt/PIC/results_mr_mr8_fadeev_harris/log_mr.txt");

const int totalStep = 160 * 10 * 10;
const int fieldRecordStep = 160 * 1;
const bool isParticleRecord = false;
const int particleRecordStep = 160 * 10;
const int particleRecordStepStart = -1;
float totalTime = 0.0f;

const float c = 1.0f;
const float epsilon0 = 1.0f;
const float mu0 = 1.0f;
const float dOfLangdonMarderTypeCorrection = 0.01f;
const float EPS = 1.0e-20f;
const float PI = 3.14159265358979; 

const int numberDensityIon = 50;
const int numberDensityElectron = 50;

const float B0 = sqrt(static_cast<float>(numberDensityElectron)) / 1.0f;

const float mRatio = 8.0f;
const float mElectron = 1.0f;
const float mIon = mRatio * mElectron;

const float tRatio = 1.0f;
const float tElectron = (B0 * B0 / 2.0 / mu0) / (numberDensityIon + numberDensityElectron * tRatio);
const float tIon = tRatio * tElectron;

const float qRatio = -1.0f;
const float qElectron = -1.0f * sqrt(epsilon0 * tElectron / static_cast<float>(numberDensityElectron)) / 1.0f;
const float qIon = qRatio * qElectron;

const float omegaPe = sqrt(static_cast<float>(numberDensityElectron) * pow(qElectron, 2) / mElectron / epsilon0);
const float omegaPi = sqrt(static_cast<float>(numberDensityIon) * pow(qIon, 2) / mIon / epsilon0);
const float omegaCe = abs(qElectron * B0 / mElectron);
const float omegaCi = qIon * B0 / mIon;

const float debyeLength = sqrt(epsilon0 * tElectron / static_cast<float>(numberDensityElectron) / pow(qElectron, 2));
//追加
const float ionInertialLength = c / omegaPi;

//追加
const float fadeevSheatThickness = 30.0f * ionInertialLength;
const float harrisSheatThickness = 3.0f * ionInertialLength;

const int nx = round(4.0f * PI * fadeevSheatThickness);
const float dx = 1.0f;
const float xmin = 0.0f * dx; 
const float xmax = nx * dx - 0.0f * dx;

const int ny = round(2.0f * PI * fadeevSheatThickness);
const float dy = 1.0f;
const float ymin = 1.0f * dy; 
const float ymax = ny * dy - 1.5f * dy;

const float triggerRatio = 0.1f;
const float xPointPosition = 0.5f * nx * dx;
const float coefFadeev = 0.7f; 

unsigned long long fadeevNumIon = round(25.13f * pow(fadeevSheatThickness, 2) * numberDensityIon);
unsigned long long fadeevNumElectron = round(25.13f * pow(fadeevSheatThickness, 2) * numberDensityElectron);
unsigned long long harrisNumIon = round(nx * numberDensityIon * 2.0f * harrisSheatThickness);
unsigned long long harrisNumElectron = round(nx * numberDensityElectron * 2.0f * harrisSheatThickness);
unsigned long long backgroundNumIon = round(0.2f * nx * ny * numberDensityIon);
unsigned long long backgroundNumElectron = round(0.2f * nx * ny * numberDensityElectron);
const unsigned long long totalNumIon = fadeevNumIon + harrisNumIon + backgroundNumIon;
const unsigned long long totalNumElectron = fadeevNumElectron + harrisNumElectron + backgroundNumElectron;
const unsigned long long totalNumParticles = totalNumIon + totalNumElectron;

const float dt = 0.5f;

const float vThIon = sqrt(2.0f * tIon / mIon);
const float vThElectron = sqrt(2.0f * tElectron / mElectron);
const float bulkVxIon = 1.0e10f;
const float bulkVyIon = 1.0e10f;
const float bulkVzIon = 1.0e10f;
const float bulkVxElectron = 1.0e10f;
const float bulkVyElectron = 1.0e10f;
const float bulkVzElectron = 1.0e10f;
const float bulkVxIonFadeev = 0.0f;
const float bulkVyIonFadeev = 0.0f;
const float bulkVzIonFadeev = -B0 / mu0 / numberDensityIon / qIon / fadeevSheatThickness / (1.0f + 1.0f / tRatio);
const float bulkVxElectronFadeev = -bulkVxIonFadeev / tRatio;
const float bulkVyElectronFadeev = -bulkVyIonFadeev / tRatio;
const float bulkVzElectronFadeev = -bulkVzIonFadeev / tRatio;
const float bulkVxIonHarris = 0.0f;
const float bulkVyIonHarris = 0.0f;
const float bulkVzIonHarris = -B0 / mu0 / numberDensityIon / qIon / harrisSheatThickness / (1.0f + 1.0f / tRatio);
const float bulkVxElectronHarris = -bulkVxIonHarris / tRatio;
const float bulkVyElectronHarris = -bulkVyIonHarris / tRatio;
const float bulkVzElectronHarris = -bulkVzIonHarris / tRatio;

const float vThIonB = sqrt(2.0f * tIon * 0.2f / mIon);
const float vThElectronB = sqrt(2.0f * tElectron * 0.2f / mElectron);
const float bulkVxElectronB = 0.0f;
const float bulkVyElectronB = 0.0f;
const float bulkVzElectronB = 0.0f;
const float bulkVxIonB = 0.0f;
const float bulkVyIonB = 0.0f;
const float bulkVzIonB = 0.0f;


__constant__ int device_totalStep;
__constant__ int device_recordStep;
__device__ float device_totalTime;

__constant__ float device_c;
__constant__ float device_epsilon0;
__constant__ float device_mu0;
__constant__ float device_dOfLangdonMarderTypeCorrection;
__constant__ float device_EPS;
__constant__ float device_PI; 

__constant__ int device_numberDensityIon;
__constant__ int device_numberDensityElectron;

__constant__ float device_B0;

__constant__ float device_mRatio;
__constant__ float device_mElectron;
__constant__ float device_mIon;

__constant__ float device_tRatio;
__constant__ float device_tElectron;
__constant__ float device_tIon;

__constant__ float device_qRatio;
__constant__ float device_qElectron;
__constant__ float device_qIon;

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
__constant__ float device_fadeevSheatThickness;
__constant__ float device_harrisSheatThickness;
__constant__ float device_triggerRatio;
__constant__ float device_xPointPosition;
__constant__ float device_coefFadeev; 

//追加
__constant__ unsigned long long device_totalNumIon;
__constant__ unsigned long long device_totalNumElectron;
__constant__ unsigned long long device_totalNumParticles;

__constant__ float device_vThIon;
__constant__ float device_vThElectron;
__constant__ float device_bulkVxElectron;
__constant__ float device_bulkVyElectron;
__constant__ float device_bulkVzElectron;
__constant__ float device_bulkVxIon;
__constant__ float device_bulkVyIon;
__constant__ float device_bulkVzIon;

__constant__ float device_vThIonB;
__constant__ float device_vThElectronB;
__constant__ float device_bulkVxElectronB;
__constant__ float device_bulkVyElectronB;
__constant__ float device_bulkVzElectronB;
__constant__ float device_bulkVxIonB;
__constant__ float device_bulkVyIonB;
__constant__ float device_bulkVzIonB;


__global__ void initializeField_kernel(
    ElectricField* E, MagneticField* B
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx && j < device_ny) {
        float xCenter = 0.5f * (device_xmax - device_xmin) + device_xmin, yCenter = 0.5f * (device_ymax - device_ymin) + device_ymin;
        float x = i * device_dx, y = j * device_dy;
        float phaseX = (x - xCenter) / device_fadeevSheatThickness, phaseY = (y - yCenter) / device_fadeevSheatThickness;
        float VA = device_B0 / sqrt(device_mu0 * (device_mIon * device_numberDensityIon + device_mElectron * device_numberDensityElectron));
        
        E[j + device_ny * i].eX = 0.0f;
        E[j + device_ny * i].eY = 0.0f;
        E[j + device_ny * i].eZ = 0.0f; 
        B[j + device_ny * i].bX = device_B0 * sinh(phaseY) / (cosh(phaseY) + device_coefFadeev * cos(phaseX))
                                + device_B0 * tanh((y - yCenter) / device_harrisSheatThickness)
                                - device_B0 * device_triggerRatio * (y - yCenter) / device_fadeevSheatThickness
                                * exp(-(pow((x - xCenter), 2) + pow((y - yCenter), 2))
                                / pow(2.0f * device_fadeevSheatThickness, 2));
        B[j + device_ny * i].bY = device_B0 * device_coefFadeev * sin(phaseX) / (cosh(phaseY) + device_coefFadeev * cos(phaseX))
                                + device_B0 * device_triggerRatio * (x - xCenter) / device_fadeevSheatThickness
                                * exp(-(pow((x - xCenter), 2) + pow((y - yCenter), 2))
                                / pow(2.0f * device_fadeevSheatThickness, 2)); 
        B[j + device_ny * i].bZ = 0.0f;
    }
}

void PIC2D::initialize()
{
    cudaMemcpyToSymbol(device_PI, &PI, sizeof(float));
    cudaMemcpyToSymbol(device_ionInertialLength, &ionInertialLength, sizeof(float));
    cudaMemcpyToSymbol(device_fadeevSheatThickness, &fadeevSheatThickness, sizeof(float));
    cudaMemcpyToSymbol(device_harrisSheatThickness, &harrisSheatThickness, sizeof(float));
    cudaMemcpyToSymbol(device_triggerRatio, &triggerRatio, sizeof(float));
    cudaMemcpyToSymbol(device_xPointPosition, &xPointPosition, sizeof(float));
    cudaMemcpyToSymbol(device_coefFadeev, &coefFadeev, sizeof(float));

    initializeParticle.fadeevForPosition(
        0, fadeevNumIon, 
        0, 
        fadeevSheatThickness, coefFadeev, 
        particlesIon
    );
    initializeParticle.fadeevForPosition(
        0, fadeevNumElectron, 
        10000, 
        fadeevSheatThickness, coefFadeev,
        particlesElectron
    );
    initializeParticle.uniformForPositionX(
        fadeevNumIon, totalNumIon, 20000, particlesIon
    );
    initializeParticle.uniformForPositionX(
        fadeevNumElectron, totalNumElectron, 30000, particlesElectron
    );
    initializeParticle.harrisForPositionY(
        fadeevNumIon, fadeevNumIon + harrisNumIon, 40000, 
        harrisSheatThickness, particlesIon
    );
    initializeParticle.harrisForPositionY(
        fadeevNumElectron, fadeevNumElectron + harrisNumElectron, 50000, 
        harrisSheatThickness, particlesElectron
    );
    initializeParticle.uniformForPositionY(
        fadeevNumIon + harrisNumIon, totalNumIon, 60000, particlesIon
    );
    initializeParticle.uniformForPositionY(
        fadeevNumElectron + harrisNumElectron, totalNumElectron, 70000, particlesElectron
    );
    
    initializeParticle.maxwellDistributionForVelocity(
        bulkVxIonFadeev, bulkVyIonFadeev, bulkVzIonFadeev, 
        vThIon, vThIon, vThIon, 
        0, fadeevNumIon, 80000, particlesIon
    );
    initializeParticle.maxwellDistributionForVelocity(
        bulkVxIonHarris, bulkVyIonHarris, bulkVzIonHarris, 
        vThIon, vThIon, vThIon, 
        fadeevNumIon, fadeevNumIon + harrisNumIon, 90000, particlesIon
    );
    initializeParticle.maxwellDistributionForVelocity(
        bulkVxIonB, bulkVyIonB, bulkVzIonB, vThIonB, vThIonB, vThIonB, 
        fadeevNumIon + harrisNumIon, totalNumIon, 100000, particlesIon
    );
    initializeParticle.maxwellDistributionForVelocity(
        bulkVxElectronFadeev, bulkVyElectronFadeev, bulkVzElectronFadeev, 
        vThElectron, vThElectron, vThElectron, 
        0, fadeevNumElectron, 110000, particlesElectron
    );
    initializeParticle.maxwellDistributionForVelocity(
        bulkVxElectronHarris, bulkVyElectronHarris, bulkVzElectronHarris, 
        vThElectron, vThElectron, vThElectron, 
        fadeevNumElectron, fadeevNumElectron + harrisNumElectron, 120000, particlesElectron
    );
    initializeParticle.maxwellDistributionForVelocity(
        bulkVxElectronB, bulkVyElectronB, bulkVzElectronB, vThElectronB, vThElectronB, vThElectronB, 
        fadeevNumElectron + harrisNumElectron, totalNumElectron, 130000, particlesElectron
    );

    boundary.periodicBoundaryParticleX(particlesIon, particlesElectron);
    boundary.conductingWallBoundaryParticleY(particlesIon, particlesElectron);


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

    std::cout << "box size is " << nx << " X " << ny << std::endl;
    std::cout << "total number of partices is " << totalNumParticles << std::endl;
    std::cout << std::setprecision(4) 
              << "Omega_ci * t = " << totalStep * dt * omegaCi << std::endl;


    PIC2D pIC2D;

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
            pIC2D.saveZerothMoments(
                directoryname, filenameWithoutStep, step
            );
            pIC2D.saveFirstMoments(
                directoryname, filenameWithoutStep, step
            );
        }
        if (isParticleRecord && (step % particleRecordStep == 0) && (step >= particleRecordStepStart)) {
            pIC2D.saveParticle(
                directoryname, filenameWithoutStep, step
            );
        }

        
        pIC2D.oneStepPeriodicXWallY();

        totalTime += dt;

        if (step == 0) {
            size_t free_mem = 0;
            size_t total_mem = 0;
            cudaError_t status = cudaMemGetInfo(&free_mem, &total_mem);

            std::cout << "Free memory: " << free_mem / (1024 * 1024) << " MB" << std::endl;
            std::cout << "Total memory: " << total_mem / (1024 * 1024) << " MB" << std::endl;
        }
    }

    return 0;
}



