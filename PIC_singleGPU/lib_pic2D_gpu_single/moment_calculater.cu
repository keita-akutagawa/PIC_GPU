#include <thrust/fill.h>
#include "moment_calculater.hpp"



void MomentCalculater::resetZerothMomentOfOneSpecies(
    thrust::device_vector<ZerothMoment>& zerothMomentOfOneSpecies
)
{
    thrust::fill(
        zerothMomentOfOneSpecies.begin(), 
        zerothMomentOfOneSpecies.end(), 
        ZerothMoment()
    );

    cudaDeviceSynchronize();
}

void MomentCalculater::resetFirstMomentOfOneSpecies(
    thrust::device_vector<FirstMoment>& firstMomentOfOneSpecies
)
{
    thrust::fill(
        firstMomentOfOneSpecies.begin(), 
        firstMomentOfOneSpecies.end(), 
        FirstMoment()
    );

    cudaDeviceSynchronize();
}

void MomentCalculater::resetSecondMomentOfOneSpecies(
    thrust::device_vector<SecondMoment>& secondMomentOfOneSpecies
)
{
    thrust::fill(
        secondMomentOfOneSpecies.begin(), 
        secondMomentOfOneSpecies.end(), 
        SecondMoment()
    );

    cudaDeviceSynchronize();
}

//////////

struct CalculateZerothMomentOfOneSpeciesFunctor {
    ZerothMoment* zerothMomentOfOneSpecies;
    const Particle* particlesSpecies;

    __device__
    void operator()(const unsigned long long& i) const {
        float cx1, cx2; 
        unsigned long long xIndex1, xIndex2;
        float xOverDx;
        float cy1, cy2; 
        unsigned long long yIndex1, yIndex2;
        float yOverDy;

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

        atomicAdd(&(zerothMomentOfOneSpecies[yIndex1 + device_ny * xIndex1].n), cx2 * cy2);
        atomicAdd(&(zerothMomentOfOneSpecies[yIndex2 + device_ny * xIndex1].n), cx2 * cy1);
        atomicAdd(&(zerothMomentOfOneSpecies[yIndex1 + device_ny * xIndex2].n), cx1 * cy2);
        atomicAdd(&(zerothMomentOfOneSpecies[yIndex2 + device_ny * xIndex2].n), cx1 * cy1);
    }
};


void MomentCalculater::calculateZerothMomentOfOneSpecies(
    thrust::device_vector<ZerothMoment>& zerothMomentOfOneSpecies, 
    const thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long totalNumSpecies
)
{
    resetZerothMomentOfOneSpecies(zerothMomentOfOneSpecies);

    CalculateZerothMomentOfOneSpeciesFunctor calculateZerothMomentOfOneSpeciesFunctor{
        thrust::raw_pointer_cast(zerothMomentOfOneSpecies.data()), 
        thrust::raw_pointer_cast(particlesSpecies.data())
    };

    thrust::for_each(
        thrust::counting_iterator<unsigned long long>(0), 
        thrust::counting_iterator<unsigned long long>(totalNumSpecies), 
        calculateZerothMomentOfOneSpeciesFunctor
    );

    cudaDeviceSynchronize();
}



struct CalculateFirstMomentOfOneSpeciesFunctor {
    FirstMoment* firstMomentOfOneSpecies;
    const Particle* particlesSpecies;

    __device__
    void operator()(const unsigned long long& i) const {
        float cx1, cx2; 
        unsigned long long xIndex1, xIndex2;
        float xOverDx;
        float cy1, cy2; 
        unsigned long long yIndex1, yIndex2;
        float yOverDy;
        float vx, vy, vz;

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

        vx = particlesSpecies[i].vx / particlesSpecies[i].gamma;
        vy = particlesSpecies[i].vy / particlesSpecies[i].gamma;
        vz = particlesSpecies[i].vz / particlesSpecies[i].gamma;

        atomicAdd(&(firstMomentOfOneSpecies[yIndex1 + device_ny * xIndex1].x), vx * cx2 * cy2);
        atomicAdd(&(firstMomentOfOneSpecies[yIndex2 + device_ny * xIndex1].x), vx * cx2 * cy1);
        atomicAdd(&(firstMomentOfOneSpecies[yIndex1 + device_ny * xIndex2].x), vx * cx1 * cy2);
        atomicAdd(&(firstMomentOfOneSpecies[yIndex2 + device_ny * xIndex2].x), vx * cx1 * cy1);

        atomicAdd(&(firstMomentOfOneSpecies[yIndex1 + device_ny * xIndex1].y), vy * cx2 * cy2);
        atomicAdd(&(firstMomentOfOneSpecies[yIndex2 + device_ny * xIndex1].y), vy * cx2 * cy1);
        atomicAdd(&(firstMomentOfOneSpecies[yIndex1 + device_ny * xIndex2].y), vy * cx1 * cy2);
        atomicAdd(&(firstMomentOfOneSpecies[yIndex2 + device_ny * xIndex2].y), vy * cx1 * cy1);

        atomicAdd(&(firstMomentOfOneSpecies[yIndex1 + device_ny * xIndex1].z), vz * cx2 * cy2);
        atomicAdd(&(firstMomentOfOneSpecies[yIndex2 + device_ny * xIndex1].z), vz * cx2 * cy1);
        atomicAdd(&(firstMomentOfOneSpecies[yIndex1 + device_ny * xIndex2].z), vz * cx1 * cy2);
        atomicAdd(&(firstMomentOfOneSpecies[yIndex2 + device_ny * xIndex2].z), vz * cx1 * cy1);
    }
};


void MomentCalculater::calculateFirstMomentOfOneSpecies(
    thrust::device_vector<FirstMoment>& firstMomentOfOneSpecies, 
    const thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long totalNumSpecies
)
{
    resetFirstMomentOfOneSpecies(firstMomentOfOneSpecies);

    CalculateFirstMomentOfOneSpeciesFunctor calculateFirstMomentOfOneSpeciesFunctor{
        thrust::raw_pointer_cast(firstMomentOfOneSpecies.data()), 
        thrust::raw_pointer_cast(particlesSpecies.data())
    };

    thrust::for_each(
        thrust::counting_iterator<unsigned long long>(0), 
        thrust::counting_iterator<unsigned long long>(totalNumSpecies), 
        calculateFirstMomentOfOneSpeciesFunctor
    );

    cudaDeviceSynchronize();
}



struct CalculateSecondMomentOfOneSpeciesFunctor {
    SecondMoment* secondMomentOfOneSpecies;
    const Particle* particlesSpecies;

    __device__
    void operator()(const unsigned long long& i) const {
        float cx1, cx2; 
        int xIndex1, xIndex2;
        float xOverDx;
        float cy1, cy2; 
        int yIndex1, yIndex2;
        float yOverDy;
        float vx, vy, vz;

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

        vx = particlesSpecies[i].vx / particlesSpecies[i].gamma;
        vy = particlesSpecies[i].vy / particlesSpecies[i].gamma;
        vz = particlesSpecies[i].vz / particlesSpecies[i].gamma;

        atomicAdd(&(secondMomentOfOneSpecies[yIndex1 + device_ny * xIndex1].xx), vx * vx * cx2 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex2 + device_ny * xIndex1].xx), vx * vx * cx2 * cy1);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex1 + device_ny * xIndex2].xx), vx * vx * cx1 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex2 + device_ny * xIndex2].xx), vx * vx * cx1 * cy1);

        atomicAdd(&(secondMomentOfOneSpecies[yIndex1 + device_ny * xIndex1].yy), vy * vy * cx2 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex2 + device_ny * xIndex1].yy), vy * vy * cx2 * cy1);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex1 + device_ny * xIndex2].yy), vy * vy * cx1 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex2 + device_ny * xIndex2].yy), vy * vy * cx1 * cy1);

        atomicAdd(&(secondMomentOfOneSpecies[yIndex1 + device_ny * xIndex1].zz), vz * vz * cx2 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex2 + device_ny * xIndex1].zz), vz * vz * cx2 * cy1);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex1 + device_ny * xIndex2].zz), vz * vz * cx1 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex2 + device_ny * xIndex2].zz), vz * vz * cx1 * cy1);

        atomicAdd(&(secondMomentOfOneSpecies[yIndex1 + device_ny * xIndex1].xy), vx * vy * cx2 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex2 + device_ny * xIndex1].xy), vx * vy * cx2 * cy1);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex1 + device_ny * xIndex2].xy), vx * vy * cx1 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex2 + device_ny * xIndex2].xy), vx * vy * cx1 * cy1);

        atomicAdd(&(secondMomentOfOneSpecies[yIndex1 + device_ny * xIndex1].xz), vx * vz * cx2 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex2 + device_ny * xIndex1].xz), vx * vz * cx2 * cy1);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex1 + device_ny * xIndex2].xz), vx * vz * cx1 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex2 + device_ny * xIndex2].xz), vx * vz * cx1 * cy1);

        atomicAdd(&(secondMomentOfOneSpecies[yIndex1 + device_ny * xIndex1].yz), vy * vz * cx2 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex2 + device_ny * xIndex1].yz), vy * vz * cx2 * cy1);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex1 + device_ny * xIndex2].yz), vy * vz * cx1 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[yIndex2 + device_ny * xIndex2].yz), vy * vz * cx1 * cy1);
    }
};


void MomentCalculater::calculateSecondMomentOfOneSpecies(
    thrust::device_vector<SecondMoment>& secondMomentOfOneSpecies, 
    const thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long totalNumSpecies
)
{
    resetSecondMomentOfOneSpecies(secondMomentOfOneSpecies);

    CalculateSecondMomentOfOneSpeciesFunctor calculateSecondMomentOfOneSpeciesFunctor{
        thrust::raw_pointer_cast(secondMomentOfOneSpecies.data()), 
        thrust::raw_pointer_cast(particlesSpecies.data())
    };

    thrust::for_each(
        thrust::counting_iterator<unsigned long long>(0), 
        thrust::counting_iterator<unsigned long long>(totalNumSpecies), 
        calculateSecondMomentOfOneSpeciesFunctor
    );

    cudaDeviceSynchronize();
}



