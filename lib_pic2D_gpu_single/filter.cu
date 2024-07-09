#include "filter.hpp"
#include <thrust/fill.h>


Filter::Filter()
    : rho(nx * ny), 
      F(nx * ny)
{
}


__global__ void calculateF_kernel(
    FilterField* F, ElectricField* E, RhoField* rho
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    //境界は0.0fに設定する
    if ((0 < i) && (i < device_nx - 1) && (0 < j) && (j < device_ny - 1)) {
        F[j + device_ny * i].F = ((E[j + device_ny * i].eX - E[j + device_ny * (i - 1)].eX) / device_dx 
                               + (E[j + device_ny * i].eY - E[j - 1 + device_ny * i].eY) / device_dy)
                               - rho[j + device_ny * i].rho / device_epsilon0;
    }
}

__global__ void correctE_kernel(
    ElectricField* E, FilterField* F, float dt
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < device_nx - 1) && (0 < j) && (j < device_ny - 1)) {
        E[j + device_ny * i].eX += device_dOfLangdonMarderTypeCorrection
                                 * (F[j + device_ny * (i + 1)].F - F[j + device_ny * i].F) / device_dx * dt;
        E[j + device_ny * i].eY += device_dOfLangdonMarderTypeCorrection
                                 * (F[j + 1 + device_ny * i].F - F[j + device_ny * i].F) / device_dy * dt;
    }
}


void Filter::langdonMarderTypeCorrection(
    thrust::device_vector<ElectricField>& E, 
    const thrust::device_vector<Particle>& particlesIon, 
    const thrust::device_vector<Particle>& particlesElectron, 
    const float dt
)
{
    calculateRho(particlesIon, particlesElectron);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateF_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(F.data()), 
        thrust::raw_pointer_cast(E.data()), 
        thrust::raw_pointer_cast(rho.data())
    );

    cudaDeviceSynchronize();

    correctE_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), 
        thrust::raw_pointer_cast(F.data()), 
        dt
    );

    cudaDeviceSynchronize();
}


void Filter::resetRho()
{
    thrust::fill(rho.begin(), rho.end(), RhoField());
}


void Filter::calculateRho(
    const thrust::device_vector<Particle>& particlesIon, 
    const thrust::device_vector<Particle>& particlesElectron
)
{
    resetRho();

    calculateRhoOfOneSpecies(particlesIon, qIon, totalNumIon);
    calculateRhoOfOneSpecies(particlesElectron, qElectron, totalNumElectron);
}


struct CalculateRhoOfOneSpeciesFunctor {
    RhoField* rho;
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

        atomicAdd(&(rho[yIndex1 + device_ny * xIndex1].rho), q * cx2 * cy2);
        atomicAdd(&(rho[yIndex2 + device_ny * xIndex1].rho), q * cx2 * cy1);
        atomicAdd(&(rho[yIndex1 + device_ny * xIndex2].rho), q * cx1 * cy2);
        atomicAdd(&(rho[yIndex2 + device_ny * xIndex2].rho), q * cx1 * cy1);
    }
};


void Filter::calculateRhoOfOneSpecies(
    const thrust::device_vector<Particle>& particlesSpecies, 
    float q, int totalNumSpecies
)
{
    CalculateRhoOfOneSpeciesFunctor calculateRhoOfOneSpeciesFunctor{
        thrust::raw_pointer_cast(rho.data()), 
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        q
    };

    thrust::for_each(
        thrust::counting_iterator<int>(0), 
        thrust::counting_iterator<int>(totalNumSpecies), 
        calculateRhoOfOneSpeciesFunctor
    );
}



