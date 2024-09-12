#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "../../PIC1D_multiGPU/pic1D.hpp"
#include "../../PIC1D_multiGPU/mpi.hpp"


std::string directoryname = "/cfca-work/akutagawakt/PIC_multiGPU/results_two_stream_electron";
std::string filenameWithoutStep = "two_stream_electron";
std::ofstream logfile("/cfca-work/akutagawakt/PIC_multiGPU/results_two_stream_electron/log_two_stream_electron.txt");


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


