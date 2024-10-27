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
    bool isExist;

    __host__ __device__
    Particle() : 
        x(0.0f), 
        y(0.0f), 
        z(0.0f), 
        vx(0.0f), 
        vy(0.0f), 
        vz(0.0f), 
        gamma(0.0f), 
        isExist(false)
        {}
    
    __host__ __device__
    Particle& operator=(const Particle& other)
    {
        if (this != &other) {
            x = other.x;
            y = other.y;
            z = other.z;
            vx = other.vx;
            vy = other.vy;
            vz = other.vz;
            gamma = other.gamma;
            isExist = other.isExist;
        }
        return *this;
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

