#include <cmath>
#include "current_calculator.hpp"
#include <thrust/fill.h>


void CurrentCalculator::resetCurrent(
    thrust::device_vector<CurrentField>& current
)
{
    thrust::fill(current.begin(), current.end(), CurrentField());
}


void CurrentCalculator::calculateCurrent(
    thrust::device_vector<CurrentField>& current, 
    const thrust::device_vector<Particle>& particlesIon, 
    const thrust::device_vector<Particle>& particlesElectron
)
{
    calculateCurrentOfOneSpecies(
        current, particlesIon, qIon, totalNumIon
    );
    calculateCurrentOfOneSpecies(
        current, particlesElectron, qElectron, totalNumElectron
    );
}


struct CalculateCurrent {
    CurrentField* current;
    const Particle* particlesSpecies;
    const double q;

    __device__
    void operator()(const int& i) const {
        double cx1, cx2; 
        int xIndex1, xIndex2;
        double xOverDx;
        double qOverGamma, qVxOverGamma, qVyOverGamma, qVzOverGamma;

        xOverDx = particlesSpecies[i].x / device_dx;

        xIndex1 = floorf(xOverDx);
        xIndex2 = xIndex1 + 1;
        xIndex2 = (xIndex2 == device_nx) ? 0 : xIndex2;

        cx1 = xOverDx - xIndex1;
        cx2 = 1.0 - cx1;

        qOverGamma = q / particlesSpecies[i].gamma;
        qVxOverGamma = qOverGamma * particlesSpecies[i].vx;
        qVyOverGamma = qOverGamma * particlesSpecies[i].vy;
        qVzOverGamma = qOverGamma * particlesSpecies[i].vz;

        atomicAdd(&(current[xIndex1].jX), qVxOverGamma * cx2);
        atomicAdd(&(current[xIndex2].jX), qVxOverGamma * cx1);

        atomicAdd(&(current[xIndex1].jY), qVyOverGamma * cx2);
        atomicAdd(&(current[xIndex2].jY), qVyOverGamma * cx1);

        atomicAdd(&(current[xIndex1].jZ), qVzOverGamma * cx2);
        atomicAdd(&(current[xIndex2].jZ), qVzOverGamma * cx1);
    }
};


void CurrentCalculator::calculateCurrentOfOneSpecies(
    thrust::device_vector<CurrentField>& current, 
    const thrust::device_vector<Particle>& particlesSpecies, 
    double q, int totalNumSpecies
)
{

    CalculateCurrent calcCurrent{
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        q
    };

    thrust::for_each(
        thrust::counting_iterator<int>(0), 
        thrust::counting_iterator<int>(totalNumSpecies), 
        calcCurrent
    );
}


