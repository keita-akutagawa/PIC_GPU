#ifndef MOMENT_STRUCT_H
#define MOMENT_STRUCT_H


struct ZerothMoment
{
    float n;

    __host__ __device__
    ZerothMoment() : 
        n(0.0f)
        {}
    
    __host__ __device__
    ZerothMoment& operator=(const ZerothMoment& other)
    {
        if (this != &other) {
            n = other.n;
        }
        return *this;
    }
};


struct FirstMoment
{
    float x;
    float y;
    float z;

    __host__ __device__
    FirstMoment() : 
        x(0.0f), 
        y(0.0f), 
        z(0.0f)
        {}
    
    __host__ __device__
    FirstMoment& operator=(const FirstMoment& other)
    {
        if (this != &other) {
            x = other.x;
            y = other.y;
            z = other.z;
        }
        return *this;
    }
};


struct SecondMoment
{
    float xx;
    float yy;
    float zz;
    float xy;
    float xz;
    float yz;

    __host__ __device__
    SecondMoment() : 
        xx(0.0f), 
        yy(0.0f), 
        zz(0.0f), 
        xy(0.0f), 
        xz(0.0f), 
        yz(0.0f)
        {}
    
    __host__ __device__
    SecondMoment& operator=(const SecondMoment& other)
    {
        if (this != &other) {
            xx = other.xx;
            yy = other.yy;
            zz = other.zz;
            xy = other.xy;
            xz = other.xz;
            yz = other.yz;
        }
        return *this;
    }
};

#endif
