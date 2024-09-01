#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "../../lib_pic1D_gpu_single/pic1D.hpp"
#include "../../lib_pic1D_gpu_single/mpi.hpp"


std::string directoryname = "/cfca-work/akutagawakt/PIC_multiGPU/results_two_stream_electron";
std::string filenameWithoutStep = "two_stream_electron";
std::ofstream logfile("/cfca-work/akutagawakt/PIC_multiGPU/results_two_stream_electron/log_two_stream_electron.txt");


const float c = 1.0f;
const float epsilon0 = 1.0f;
const float mu0 = 1.0f;

const int nx = 512;
const float dx = 1.0f;
const float xmin = 0.0f; 
const float xmax = nx * dx;

const float dt = 0.5f;

const int numberDensityIon = 100;
const int numberDensityElectron = 100;

const int totalNumIon = nx * numberDensityIon;
//追加
const int totalNumElectronBeam1 = nx * numberDensityElectron / 2;
const int totalNumElectronBeam2 = nx * numberDensityElectron / 2;
const int totalNumElectron = totalNumElectronBeam1 + totalNumElectronBeam2;
const int totalNumParticles = totalNumIon + totalNumElectron;

const float B0 = sqrt(static_cast<float>(numberDensityElectron)) / 10.0f;

const float mRatio = 100.0f;
const float mElectron = 1.0f;
const float mIon = mRatio * mElectron;

const float tRatio = 100.0f;
const float tElectron = 0.5f * mElectron * pow(0.01f * c, 2);
const float tIon = tRatio * tElectron;

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
const float bulkVxElectron = -10.0f * vThIon;
const float bulkVyElectron = 0.0f;
const float bulkVzElectron = 0.0f;
//追加
const float bulkVxElectronBeam = 10.0f * vThIon;
const float bulkVyElectronBeam = 0.0f;
const float bulkVzElectronBeam = 0.0f;

const int totalStep = 10000;
const int recordStep = 100;
float totalTime = 0.0f;


__constant__ float device_c;
__constant__ float device_epsilon0;
__constant__ float device_mu0;

__constant__ int device_nx;
__constant__ float device_dx;
__constant__ float device_xmin; 
__constant__ float device_xmax;

__constant__ float device_dt;

__constant__ int device_numberDensityIon;
__constant__ int device_numberDensityElectron;

__constant__ int device_totalNumIon;
//追加
__constant__ int device_totalNumElectronBeam1;
__constant__ int device_totalNumElectronBeam2;
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
//追加
__constant__ float device_bulkVxElectronBeam;
__constant__ float device_bulkVyElectronBeam;
__constant__ float device_bulkVzElectronBeam;

__constant__ int device_totalStep;
__device__ float device_totalTime;


