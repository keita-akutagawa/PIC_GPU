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
    const float q;

    __device__
    void operator()(const int& i) const {
        float cx1, cx2; 
        int xIndex1, xIndex2;
        float xOverDx;
        float cy1, cy2; 
        int yIndex1, yIndex2;
        float yOverDy;
        float qOverGamma, qVxOverGamma, qVyOverGamma, qVzOverGamma;

        xOverDx = particlesSpecies[i].x / device_dx;
        yOverDy = particlesSpecies[i].y / device_dy;

        xIndex1 = floorf(xOverDx);
        xIndex2 = xIndex1 + 1;
        xIndex2 = (xIndex2 == device_nx) ? 0 : xIndex2;
        yIndex1 = floorf(yOverDy);
        yIndex2 = yIndex1 + 1;
        yIndex2 = (yIndex2 == device_ny) ? 0 : yIndex2;

        cx1 = xOverDx - xIndex1;
        cx2 = 1.0f - cx1;
        cy1 = yOverDy - yIndex1;
        cy2 = 1.0f - cy1;

        qOverGamma = q / particlesSpecies[i].gamma;
        qVxOverGamma = qOverGamma * particlesSpecies[i].vx;
        qVyOverGamma = qOverGamma * particlesSpecies[i].vy;
        qVzOverGamma = qOverGamma * particlesSpecies[i].vz;

        atomicAdd(&(current[yIndex1 + device_ny * xIndex1].jX), qVxOverGamma * cx2 * cy2);
        atomicAdd(&(current[yIndex2 + device_ny * xIndex1].jX), qVxOverGamma * cx2 * cy1);
        atomicAdd(&(current[yIndex1 + device_ny * xIndex2].jX), qVxOverGamma * cx1 * cy2);
        atomicAdd(&(current[yIndex2 + device_ny * xIndex2].jX), qVxOverGamma * cx1 * cy1);

        atomicAdd(&(current[yIndex1 + device_ny * xIndex1].jY), qVyOverGamma * cx2 * cy2);
        atomicAdd(&(current[yIndex2 + device_ny * xIndex1].jY), qVyOverGamma * cx2 * cy1);
        atomicAdd(&(current[yIndex1 + device_ny * xIndex2].jY), qVyOverGamma * cx1 * cy2);
        atomicAdd(&(current[yIndex2 + device_ny * xIndex2].jY), qVyOverGamma * cx1 * cy1);

        atomicAdd(&(current[yIndex1 + device_ny * xIndex1].jZ), qVzOverGamma * cx2 * cy2);
        atomicAdd(&(current[yIndex2 + device_ny * xIndex1].jZ), qVzOverGamma * cx2 * cy1);
        atomicAdd(&(current[yIndex1 + device_ny * xIndex2].jZ), qVzOverGamma * cx1 * cy2);
        atomicAdd(&(current[yIndex2 + device_ny * xIndex2].jZ), qVzOverGamma * cx1 * cy1);
    }
};


void CurrentCalculator::calculateCurrentOfOneSpecies(
    thrust::device_vector<CurrentField>& current, 
    const thrust::device_vector<Particle>& particlesSpecies, 
    float q, int totalNumSpecies
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


