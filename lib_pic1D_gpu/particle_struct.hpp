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
    double bx;
    double by;
    double bz;
    double ex;
    double ey; 
    double ez;

    ParticleField() : 
        bx(0.0), 
        by(0.0), 
        bz(0.0), 
        ex(0.0), 
        ey(0.0), 
        ez(0.0)
        {}
};

#endif

