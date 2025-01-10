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
    
    __host__ __device__
    MagneticField& operator=(const MagneticField& other)
    {
        if (this != &other) {
            bX = other.bX;
            bY = other.bY;
            bZ = other.bZ;
        }
        return *this;
    }
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
    
    __host__ __device__
    ElectricField& operator=(const ElectricField& other)
    {
        if (this != &other) {
            eX = other.eX;
            eY = other.eY;
            eZ = other.eZ;
        }
        return *this;
    }
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
    
    __host__ __device__
    CurrentField& operator=(const CurrentField& other)
    {
        if (this != &other) {
            jX = other.jX;
            jY = other.jY;
            jZ = other.jZ;
        }
        return *this;
    }
};


struct RhoField
{
    float rho; 

    __host__ __device__
    RhoField() : 
        rho(0.0f)
        {}
    
    __host__ __device__
    RhoField& operator=(const RhoField& other)
    {
        if (this != &other) {
            rho = other.rho;
        }
        return *this;
    }
};


struct FilterField
{
    float F;

    __host__ __device__
    FilterField() : 
        F(0.0f)
        {}
    
    __host__ __device__
    FilterField& operator=(const FilterField& other)
    {
        if (this != &other) {
            F = other.F;
        }
        return *this;
    }
};

#endif
