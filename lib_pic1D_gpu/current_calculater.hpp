#include <thrust/device_vector.h>
#include "const.hpp"
#include "particle_struct.hpp"
#include "field_parameter_struct.hpp"


class CurrentCalculater
{
private: 

public: 
    void resetCurrent(
        thrust::device_vector<CurrentField>& current
    );

    void calculateCurrent(
        thrust::device_vector<CurrentField>& current, 
        const thrust::device_vector<Particle>& particlesIon, 
        const thrust::device_vector<Particle>& particlesEleectron
    );

private:

    void calculateCurrentOfOneSpecies(
        thrust::device_vector<CurrentField>& current, 
        const thrust::device_vector<Particle>& particlesSpecies, 
        double q, int totalNumSpecies
    );
};

