#include "const.hpp"


void initializeDeviceConstants()
{
    cudaMemcpyToSymbol(device_c, &c, sizeof(float));
    cudaMemcpyToSymbol(device_epsilon0, &epsilon0, sizeof(float));
    cudaMemcpyToSymbol(device_mu0, &mu0, sizeof(float));
    cudaMemcpyToSymbol(device_dOfLangdonMarderTypeCorrection, &dOfLangdonMarderTypeCorrection, sizeof(float));

    cudaMemcpyToSymbol(device_nx, &nx, sizeof(int));
    cudaMemcpyToSymbol(device_dx, &dx, sizeof(float));
    cudaMemcpyToSymbol(device_xmin, &xmin, sizeof(float));
    cudaMemcpyToSymbol(device_xmax, &xmax, sizeof(float));

    cudaMemcpyToSymbol(device_ny, &ny, sizeof(int));
    cudaMemcpyToSymbol(device_dy, &dy, sizeof(float));
    cudaMemcpyToSymbol(device_ymin, &ymin, sizeof(float));
    cudaMemcpyToSymbol(device_ymax, &ymax, sizeof(float));

    cudaMemcpyToSymbol(device_dt, &dt, sizeof(float));

    cudaMemcpyToSymbol(device_numberDensityIon, &numberDensityIon, sizeof(int));
    cudaMemcpyToSymbol(device_numberDensityElectron, &numberDensityElectron, sizeof(int));
    cudaMemcpyToSymbol(device_numberDensityHeavyIon, &numberDensityHeavyIon, sizeof(int));

    cudaMemcpyToSymbol(device_totalNumIon, &totalNumIon, sizeof(long long));
    cudaMemcpyToSymbol(device_totalNumElectron, &totalNumElectron, sizeof(long long));
    cudaMemcpyToSymbol(device_totalNumHeavyIon, &totalNumHeavyIon, sizeof(long long));
    cudaMemcpyToSymbol(device_totalNumParticles, &totalNumParticles, sizeof(long long));

    cudaMemcpyToSymbol(device_B0, &B0, sizeof(float));

    cudaMemcpyToSymbol(device_mRatio, &mRatio, sizeof(float));
    cudaMemcpyToSymbol(device_mIon, &mIon, sizeof(float));
    cudaMemcpyToSymbol(device_mElectron, &mElectron, sizeof(float));
    cudaMemcpyToSymbol(device_mHeavyIon, &mHeavyIon, sizeof(float));

    cudaMemcpyToSymbol(device_tRatio, &tRatio, sizeof(float));
    cudaMemcpyToSymbol(device_tIon, &tIon, sizeof(float));
    cudaMemcpyToSymbol(device_tElectron, &tElectron, sizeof(float));
    cudaMemcpyToSymbol(device_tHeavyIon, &tHeavyIon, sizeof(float));

    cudaMemcpyToSymbol(device_qRatio, &qRatio, sizeof(float));
    cudaMemcpyToSymbol(device_qIon, &qIon, sizeof(float));
    cudaMemcpyToSymbol(device_qElectron, &qElectron, sizeof(float));
    cudaMemcpyToSymbol(device_qHeavyIon, &qHeavyIon, sizeof(float));

    cudaMemcpyToSymbol(device_omegaPe, &omegaPe, sizeof(float));
    cudaMemcpyToSymbol(device_omegaPi, &omegaPi, sizeof(float));
    cudaMemcpyToSymbol(device_omegaCe, &omegaCe, sizeof(float));
    cudaMemcpyToSymbol(device_omegaCi, &omegaCi, sizeof(float));

    cudaMemcpyToSymbol(device_debyeLength, &debyeLength, sizeof(float));

    cudaMemcpyToSymbol(device_vThIon, &vThIon, sizeof(float));
    cudaMemcpyToSymbol(device_vThElectron, &vThElectron, sizeof(float));
    cudaMemcpyToSymbol(device_vThHeavyIon, &vThHeavyIon, sizeof(float));

    cudaMemcpyToSymbol(device_bulkVxIon, &bulkVxIon, sizeof(float));
    cudaMemcpyToSymbol(device_bulkVyIon, &bulkVyIon, sizeof(float));
    cudaMemcpyToSymbol(device_bulkVzIon, &bulkVzIon, sizeof(float));
    cudaMemcpyToSymbol(device_bulkVxElectron, &bulkVxElectron, sizeof(float));
    cudaMemcpyToSymbol(device_bulkVyElectron, &bulkVyElectron, sizeof(float));
    cudaMemcpyToSymbol(device_bulkVzElectron, &bulkVzElectron, sizeof(float));

    cudaMemcpyToSymbol(device_totalStep, &totalStep, sizeof(int));
    cudaMemcpyToSymbol(device_totalTime, &totalTime, sizeof(float));
}

