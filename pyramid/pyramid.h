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

class pyramid : public conv2D , public virtual rgbdImage {
    public:

    ~pyramid();
    
    void downPyramid(float *src, int scale);
    
    float* dstBlur;
};

}  // namespace DynaMap