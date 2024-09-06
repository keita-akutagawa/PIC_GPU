#include "const.hpp"


void initializeDeviceConstants()
{
    cudaMemcpyToSymbol(device_c, &c, sizeof(double));
    cudaMemcpyToSymbol(device_epsilon0, &epsilon0, sizeof(double));
    cudaMemcpyToSymbol(device_mu0, &mu0, sizeof(double));

    cudaMemcpyToSymbol(device_nx, &nx, sizeof(int));
    cudaMemcpyToSymbol(device_dx, &dx, sizeof(double));
    cudaMemcpyToSymbol(device_xmin, &xmin, sizeof(double));
    cudaMemcpyToSymbol(device_xmax, &xmax, sizeof(double));

    cudaMemcpyToSymbol(device_dt, &dt, sizeof(double));

    cudaMemcpyToSymbol(device_numberDensityIon, &numberDensityIon, sizeof(int));
    cudaMemcpyToSymbol(device_numberDensityElectron, &numberDensityElectron, sizeof(int));

    cudaMemcpyToSymbol(device_totalNumIon, &totalNumIon, sizeof(int));
    cudaMemcpyToSymbol(device_totalNumElectron, &totalNumElectron, sizeof(int));
    cudaMemcpyToSymbol(device_totalNumParticles, &totalNumParticles, sizeof(int));

    cudaMemcpyToSymbol(device_B0, &B0, sizeof(double));

    cudaMemcpyToSymbol(device_mRatio, &mRatio, sizeof(double));
    cudaMemcpyToSymbol(device_mIon, &mIon, sizeof(double));
    cudaMemcpyToSymbol(device_mElectron, &mElectron, sizeof(double));

    cudaMemcpyToSymbol(device_tRatio, &tRatio, sizeof(double));
    cudaMemcpyToSymbol(device_tIon, &tIon, sizeof(double));
    cudaMemcpyToSymbol(device_tElectron, &tElectron, sizeof(double));

    cudaMemcpyToSymbol(device_qRatio, &qRatio, sizeof(double));
    cudaMemcpyToSymbol(device_qIon, &qIon, sizeof(double));
    cudaMemcpyToSymbol(device_qElectron, &qElectron, sizeof(double));

    cudaMemcpyToSymbol(device_omegaPe, &omegaPe, sizeof(double));
    cudaMemcpyToSymbol(device_omegaPi, &omegaPi, sizeof(double));
    cudaMemcpyToSymbol(device_omegaCe, &omegaCe, sizeof(double));
    cudaMemcpyToSymbol(device_omegaCi, &omegaCi, sizeof(double));

    cudaMemcpyToSymbol(device_debyeLength, &debyeLength, sizeof(double));

    cudaMemcpyToSymbol(device_vThIon, &vThIon, sizeof(double));
    cudaMemcpyToSymbol(device_vThElectron, &vThElectron, sizeof(double));
    cudaMemcpyToSymbol(device_bulkVxIon, &bulkVxIon, sizeof(double));
    cudaMemcpyToSymbol(device_bulkVyIon, &bulkVyIon, sizeof(double));
    cudaMemcpyToSymbol(device_bulkVzIon, &bulkVzIon, sizeof(double));
    cudaMemcpyToSymbol(device_bulkVxElectron, &bulkVxElectron, sizeof(double));
    cudaMemcpyToSymbol(device_bulkVyElectron, &bulkVyElectron, sizeof(double));
    cudaMemcpyToSymbol(device_bulkVzElectron, &bulkVzElectron, sizeof(double));

    cudaMemcpyToSymbol(device_totalStep, &totalStep, sizeof(int));
    cudaMemcpyToSymbol(device_totalTime, &totalTime, sizeof(double));
}

