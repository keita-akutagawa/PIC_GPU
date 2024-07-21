#ifndef CONST_STRUCT_H
#define CONST_STRUCT_H


extern const float c;
extern const float epsilon0;
extern const float mu0;
extern const float dOfLangdonMarderTypeCorrection;

extern const int nx;
extern const float dx;
extern const float xmin; 
extern const float xmax;

extern const int ny;
extern const float dy;
extern const float ymin; 
extern const float ymax;

extern const float dt;

extern const int numberDensityIon;
extern const int numberDensityElectron;
extern const int numberDensityHeavyIon;

extern const long long totalNumIon;
extern const long long totalNumElectron;
extern const long long totalNumHeavyIon;
extern const long long totalNumParticles;

extern const float B0;

extern const float mRatio;
extern const float mIon;
extern const float mElectron;
extern const float mHeavyIon;

extern const float tRatio;
extern const float tIon;
extern const float tElectron;
extern const float tHeavyIon;

extern const float qRatio;
extern const float qIon;
extern const float qElectron;
extern const float qHeavyIon;

extern const float omegaPe;
extern const float omegaPi;
extern const float omegaCe;
extern const float omegaCi;

extern const float debyeLength;

extern const float vThIon;
extern const float vThElectron;
extern const float vThHeavyIon;

extern const float bulkVxIon;
extern const float bulkVyIon;
extern const float bulkVzIon;
extern const float bulkVxElectron;
extern const float bulkVyElectron;
extern const float bulkVzElectron;

extern const int totalStep;
extern float totalTime;



extern __constant__ float device_c;
extern __constant__ float device_epsilon0;
extern __constant__ float device_mu0;
extern __constant__ float device_dOfLangdonMarderTypeCorrection;

extern __constant__ int device_nx;
extern __constant__ float device_dx;
extern __constant__ float device_xmin; 
extern __constant__ float device_xmax;

extern __constant__ int device_ny;
extern __constant__ float device_dy;
extern __constant__ float device_ymin; 
extern __constant__ float device_ymax;

extern __constant__ float device_dt;

extern __constant__ int device_numberDensityIon;
extern __constant__ int device_numberDensityElectron;
extern __constant__ int device_numberDensityHeavyIon;

extern __constant__ long long device_totalNumIon;
extern __constant__ long long device_totalNumElectron;
extern __constant__ long long device_totalNumHeavyIon;
extern __constant__ long long device_totalNumParticles;

extern __constant__ float device_B0;

extern __constant__ float device_mRatio;
extern __constant__ float device_mIon;
extern __constant__ float device_mElectron;
extern __constant__ float device_mHeavyIon;

extern __constant__ float device_tRatio;
extern __constant__ float device_tIon;
extern __constant__ float device_tElectron;
extern __constant__ float device_tHeavyIon;

extern __constant__ float device_qRatio;
extern __constant__ float device_qIon;
extern __constant__ float device_qElectron;
extern __constant__ float device_qHeavyIon;

extern __constant__ float device_omegaPe;
extern __constant__ float device_omegaPi;
extern __constant__ float device_omegaCe;
extern __constant__ float device_omegaCi;

extern __constant__ float device_debyeLength;

extern __constant__ float device_vThIon;
extern __constant__ float device_vThElectron;
extern __constant__ float device_vThHeavyIon;

extern __constant__ float device_bulkVxIon;
extern __constant__ float device_bulkVyIon;
extern __constant__ float device_bulkVzIon;
extern __constant__ float device_bulkVxElectron;
extern __constant__ float device_bulkVyElectron;
extern __constant__ float device_bulkVzElectron;
extern __constant__ float device_bulkVxHeavyIon;
extern __constant__ float device_bulkVyHeavyIon;
extern __constant__ float device_bulkVzHeavyIon;

extern __constant__ int device_totalStep;
extern __device__ float device_totalTime;


void initializeDeviceConstants();

#endif
