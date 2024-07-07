#ifndef PARTICLE_STRUCT_H
#define PARTICLE_STRUCT_H

struct Particle
{
    double x;
    double y;
    double z;
    double vx;
    double vy; 
    double vz;
    double gamma;

    __host__ __device__
    Particle() : 
        x(0.0), 
        y(0.0), 
        z(0.0), 
        vx(0.0), 
        vy(0.0), 
        vz(0.0), 
        gamma(0.0)
        {}
};


struct ParticleField
{
    double bX;
    double bY;
    double bZ;
    double eX;
    double eY; 
    double eZ;

    __host__ __device__
    ParticleField() : 
        bX(0.0), 
        bY(0.0), 
        bZ(0.0), 
        eX(0.0), 
        eY(0.0), 
        eZ(0.0)
        {}
};

#endif

