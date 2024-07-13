#ifndef PARTICLE_STRUCT_H
#define PARTICLE_STRUCT_H

#include "const.hpp"

struct Particle
{
    float x;
    float y;
    float z;
    float vx;
    float vy; 
    float vz;
    float gamma;

    __host__ __device__
    Particle() : 
        x(0.0f), 
        y(0.0f), 
        z(0.0f), 
        vx(0.0f), 
        vy(0.0f), 
        vz(0.0f), 
        gamma(0.0f)
        {}
    
    __device__
    bool operator<(const Particle& other) const
    {
        //return int(y) + device_ny * int(x) < int(other.y) + device_ny * int(other.x);
        return y < other.y;
    }
};


struct ParticleField
{
    float bX;
    float bY;
    float bZ;
    float eX;
    float eY; 
    float eZ;

    __host__ __device__
    ParticleField() : 
        bX(0.0f), 
        bY(0.0f), 
        bZ(0.0f), 
        eX(0.0f), 
        eY(0.0f), 
        eZ(0.0f)
        {}
};

#endif

