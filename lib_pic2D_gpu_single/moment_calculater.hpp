#include <thrust/device_vector.h>
#include "moment_struct.hpp"
#include "const.hpp"


class MomentCalculater
{
private:

public:
    void calculateZerothMoment(
        thrust::device_vector<ZerothMoment>& zerothMomentIon, 
        thrust::device_vector<ZerothMoment>& zerothMomentElectron
    );

    void calculateFirstMoment(
        thrust::device_vector<FirstMoment>& firstMomentIon, 
        thrust::device_vector<FirstMoment>& firstMomentElectron
    );

    void calculateSecondMoment(
        thrust::device_vector<SecondMoment>& secondMomentIon, 
        thrust::device_vector<SecondMoment>& secondMomentElectron
    );

private:

};



