#ifndef FIELD_STRUCT_H
#define FIELD_STRUCT_H

struct MagneticField
{
    double bX; 
    double bY; 
    double bZ; 

    __host__ __device__
    MagneticField() : 
        bX(0.0),
        bY(0.0),
        bZ(0.0)
        {}
};


struct ElectricField
{
    double eX; 
    double eY; 
    double eZ; 

    __host__ __device__
    ElectricField() : 
        eX(0.0),
        eY(0.0),
        eZ(0.0)
        {}
};


struct CurrentField
{
    double jX; 
    double jY; 
    double jZ; 

    __host__ __device__
    CurrentField() : 
        jX(0.0),
        jY(0.0),
        jZ(0.0)
        {}
};

#endif