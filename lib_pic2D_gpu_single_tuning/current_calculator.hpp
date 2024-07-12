#include <thrust/device_vector.h>
#include "const.hpp"
#include "particle_struct.hpp"
#include "field_parameter_struct.hpp"


class CurrentCalculator
{
private: 

public: 
    void resetCurrent(
        thrust::device_vector<CurrentField>& current
    );

    void calculateCurrent(
        thrust::device_vector<CurrentField>& current, 
        const thrust::device_vector<Particle>& particlesIon, 
        const thrust::device_vector<Particle>& particlesEleectron, 
        const thrust::device_vector<unsigned long long>& particleIndexIon, 
        const thrust::device_vector<unsigned long long>& particleIndexElectron
    );

private:

    void calculateCurrentOfOneSpecies(
        thrust::device_vector<CurrentField>& current, 
        const thrust::device_vector<Particle>& particlesSpecies, 
        const thrust::device_vector<unsigned long long>& particlesIndexSpecies, 
        float q, int totalNumSpecies
    );
};

