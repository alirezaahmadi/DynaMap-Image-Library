/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% by: Alireza Ahmadi                                     %
% University of Bonn- MSc Robotics & Geodetic Engineering%
% Alireza.Ahmadi@uni-bonn.de                             %
% AlirezaAhmadi.xyz                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "conv2D/conv2D.h"
#include "utils.h"

namespace DynaMap{

class filter2D : public conv2D , public virtual rgbdImage {
    public:

    ~filter2D();
    
    void bilateralFilter(float* outImage, float* inImage, const gaussianKernal& kernel);

};

}  // namespace DynaMap