#ifndef MOMENT_STRUCT_H
#define MOMENT_STRUCT_H


struct ZerothMoment
{
    double n;

    __host__ __device__
    ZerothMoment() : 
        n(0.0)
        {}
};


struct FirstMoment
{
    double x;
    double y;
    double z;

    __host__ __device__
    FirstMoment() : 
        x(0.0), 
        y(0.0), 
        z(0.0)
        {}
};


struct SecondMoment
{
    double xx;
    double yy;
    double zz;
    double xy;
    double xz;
    double yz;

    __host__ __device__
    SecondMoment() : 
        xx(0.0), 
        yy(0.0), 
        zz(0.0), 
        xy(0.0), 
        xz(0.0), 
        yz(0.0)
        {}
};

#endif