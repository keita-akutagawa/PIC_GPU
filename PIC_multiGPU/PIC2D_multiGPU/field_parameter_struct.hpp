#ifndef FIELD_STRUCT_H
#define FIELD_STRUCT_H

struct MagneticField
{
    float bX; 
    float bY; 
    float bZ; 

    __host__ __device__
    MagneticField() : 
        bX(0.0f),
        bY(0.0f),
        bZ(0.0f)
        {}
};


struct ElectricField
{
    float eX; 
    float eY; 
    float eZ; 

    __host__ __device__
    ElectricField() : 
        eX(0.0f),
        eY(0.0f),
        eZ(0.0f)
        {}
};


struct CurrentField
{
    float jX; 
    float jY; 
    float jZ; 

    __host__ __device__
    CurrentField() : 
        jX(0.0f),
        jY(0.0f),
        jZ(0.0f)
        {}
};


struct RhoField
{
    float rho; 

    __host__ __device__
    RhoField() : 
        rho(0.0f)
        {}
};


struct FilterField
{
    float F;

    __host__ __device__
    FilterField() : 
        F(0.0f)
        {}
};

#endif
