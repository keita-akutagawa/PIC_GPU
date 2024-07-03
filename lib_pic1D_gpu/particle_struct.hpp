#ifndef PARTICLE_STRUCT_H
#define PARTICLE_STRUCT_H

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
};


struct ParticleField
{
    float bx;
    float by;
    float bz;
    float ex;
    float ey; 
    float ez;

    __host__ __device__
    ParticleField() : 
        bx(0.0f), 
        by(0.0f), 
        bz(0.0f), 
        ex(0.0f), 
        ey(0.0f), 
        ez(0.0f)
        {}
};

#endif

