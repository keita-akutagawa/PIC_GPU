#include <thrust/device_vector.h>
#include "const.hpp"
#include "field_parameter_struct.hpp"
#include "particle_struct.hpp"


struct FilterField
{
    float F;

    __host__ __device__
    FilterField() : 
        F(0.0f)
        {}
};


class Filter
{
private:

    thrust::device_vector<RhoField> rho;
    thrust::device_vector<FilterField> F;

public:
    Filter();

    void langdonMarderTypeCorrection(
        thrust::device_vector<ElectricField>& E, 
        const thrust::device_vector<Particle>& particlesIon, 
        const thrust::device_vector<Particle>& particlesElectron, 
        const thrust::device_vector<Particle>& particlesHeavyIon, 
        const float dt
    );

private:
    void resetRho();

    void calculateRho(
        const thrust::device_vector<Particle>& particlesIon, 
        const thrust::device_vector<Particle>& particlesElectron, 
        const thrust::device_vector<Particle>& particlesHeavyIon
    );

    void calculateRhoOfOneSpecies(
        const thrust::device_vector<Particle>& particlesSpecies, 
        float q, int totalNumSpecies
    );
};




