#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "../../PIC2D_multiGPU/pic2D.hpp"
#include "../../PIC2D_multiGPU/mpi.hpp"
#include <cuda_runtime.h>


std::string directoryname = "/cfca-work/akutagawakt/PIC_multiGPU/results_weibel";
std::string filenameWithoutStep = "weibel";
std::ofstream logfile("/cfca-work/akutagawakt/PIC_multiGPU/results_weibel/log_weibel.txt");
std::ofstream mpifile("/cfca-work/akutagawakt/PIC_multiGPU/results_weibel/mpi_weibel.txt");

const int buffer = 2;

const int totalStep = 3000;
const int fieldRecordStep = 100;
const bool isParticleRecord = true;
const int particleRecordStep = 1000000;
float totalTime = 0.0f;

const float c = 1.0f;
const float epsilon0 = 1.0f;
const float mu0 = 1.0f;
const float dOfLangdonMarderTypeCorrection = 0.0f;
const float EPS = 1.0e-10f;

const int numberDensityIon = 100;
const int numberDensityElectron = 100;

const float B0 = 0.0f;

const float mRatio = 1.0f;
const float mElectron = 1.0f;
const float mIon = mRatio * mElectron;

const float tRatio = 1.0f;
const float tElectron = 0.5f * mElectron * pow(0.1f * c, 2);
const float tIon = tRatio * tElectron;

const float qRatio = -1.0f;
const float qElectron = -1.0f * sqrt(epsilon0 * tElectron / static_cast<float>(numberDensityElectron)) / 1.0f;
const float qIon = qRatio * qElectron;

const float omegaPe = sqrt(static_cast<float>(numberDensityElectron) * pow(qElectron, 2) / mElectron / epsilon0);
const float omegaPi = sqrt(static_cast<float>(numberDensityIon) * pow(qIon, 2) / mIon / epsilon0);
const float omegaCe = abs(qElectron * B0 / mElectron);
const float omegaCi = qIon * B0 / mIon;

const float debyeLength = sqrt(epsilon0 * tElectron / static_cast<float>(numberDensityElectron) / pow(qElectron, 2));
const float ionInertialLength = c / omegaPi;

const int nx = 256;
const float dx = 1.0f;
const float xmin = EPS; 
const float xmax = nx * dx - EPS;

const int ny = 256;
const float dy = 1.0f;
const float ymin = EPS; 
const float ymax = ny * dy - EPS;

float dt = 0.5f;

const unsigned long long totalNumIon = numberDensityIon * nx * ny;
const unsigned long long totalNumElectron = numberDensityElectron * nx * ny;
const unsigned long long totalNumParticles = totalNumIon + totalNumElectron;

unsigned long long existNumIonPerProcs;
unsigned long long existNumElectronPerProcs;
unsigned long long totalNumIonPerProcs;
unsigned long long totalNumElectronPerProcs;

float xminForProcs;
float xmaxForProcs;
float yminForProcs;
float ymaxForProcs;

const float vThIon = sqrt(2.0f * tIon / mIon);
const float vThElectron = sqrt(2.0f * tElectron / mElectron);
const float bulkVxIon = 0.0f;
const float bulkVyIon = 0.0f;
const float bulkVzIon = 0.0f;
const float bulkVxElectron = 0.0f;
const float bulkVyElectron = 0.0f;
const float bulkVzElectron = 0.0f;


__constant__ int device_totalStep;
__device__ float device_totalTime;

__constant__ float device_c;
__constant__ float device_epsilon0;
__constant__ float device_mu0;
__constant__ float device_dOfLangdonMarderTypeCorrection;
__constant__ float device_EPS;

__constant__ int device_numberDensityIon;
__constant__ int device_numberDensityElectron;

__constant__ unsigned long long device_totalNumIon;
__constant__ unsigned long long device_totalNumElectron;
__constant__ unsigned long long device_totalNumParticles;

__device__ unsigned long long device_existNumIonPerProcs;
__device__ unsigned long long device_existNumElectronPerProcs;
__device__ unsigned long long device_totalNumIonPerProcs;
__device__ unsigned long long device_totalNumElectronPerProcs;

__device__ float device_xminForProcs;
__device__ float device_xmaxForProcs;
__device__ float device_yminForProcs;
__device__ float device_ymaxForProcs;

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
__constant__ float device_ionInertialLength;

__constant__ int device_nx;
__constant__ float device_dx;
__constant__ float device_xmin;
__constant__ float device_xmax;

__constant__ int device_ny;
__constant__ float device_dy;
__constant__ float device_ymin;
__constant__ float device_ymax;

__device__ float device_dt;

__constant__ float device_vThIon;
__constant__ float device_vThElectron;
__constant__ float device_bulkVxIon;
__constant__ float device_bulkVyIon;
__constant__ float device_bulkVzIon;
__constant__ float device_bulkVxElectron;
__constant__ float device_bulkVyElectron;
__constant__ float device_bulkVzElectron;


