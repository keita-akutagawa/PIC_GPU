#include "boundary.hpp"



//////////

void Boundary::periodicBoundaryB_x(
    thrust::device_vector<MagneticField>& B
)
{
    sendrecv_magneticField_x(
        B, 
        sendMagneticFieldLeft, sendMagneticFieldRight, 
        recvMagneticFieldLeft, recvMagneticFieldRight, 
        mPIInfo
    ); 
}


void Boundary::periodicBoundaryB_y(
    thrust::device_vector<MagneticField>& B
)
{
    sendrecv_magneticField_y(
        B, 
        sendMagneticFieldDown, sendMagneticFieldUp, 
        recvMagneticFieldDown, recvMagneticFieldUp, 
        mPIInfo
    ); 
}

//////////

void Boundary::periodicBoundaryE_x(
    thrust::device_vector<ElectricField>& E
)
{
    sendrecv_electricField_x(
        E, 
        sendElectricFieldLeft, sendElectricFieldRight, 
        recvElectricFieldLeft, recvElectricFieldRight, 
        mPIInfo
    ); 
}


void Boundary::periodicBoundaryE_y(
    thrust::device_vector<ElectricField>& E
)
{
    sendrecv_electricField_y(
        E, 
        sendElectricFieldDown, sendElectricFieldUp, 
        recvElectricFieldDown, recvElectricFieldUp, 
        mPIInfo
    ); 
}

//////////

void Boundary::periodicBoundaryCurrent_x(
    thrust::device_vector<CurrentField>& current
)
{
    sendrecv_currentField_x(
        current, 
        sendCurrentFieldLeft, sendCurrentFieldRight, 
        recvCurrentFieldLeft, recvCurrentFieldRight, 
        mPIInfo
    ); 
}


void Boundary::periodicBoundaryCurrent_y(
    thrust::device_vector<CurrentField>& current
)
{
    sendrecv_currentField_y(
        current, 
        sendCurrentFieldDown, sendCurrentFieldUp, 
        recvCurrentFieldDown, recvCurrentFieldUp, 
        mPIInfo
    ); 
}



