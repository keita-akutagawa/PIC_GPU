#include "../const.hpp"


const int nx = 100;
const double dx = 1.0;
const double dt = 0.5;

const int numberDensityIon = 10;
const int numberDensityElectron = 10;

const int totalNumIon = numberDensityIon * nx;
const int totalNumElectron = numberDensityElectron * nx;

const int totalNumParticles = totalNumIon + totalNumElectron;

const double c = 1.0;
const double epsilon0 = 1.0;
const double mu0 = 1.0;

const double mIon = 100;
const double mElectron = 1.0;
const double qIon = 1.0;
const double qElectron = -1.0;

const int totalStep = 100;
const double totalTime = 0.0;
