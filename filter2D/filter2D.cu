/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% by: Alireza Ahmadi                                     %
% University of Bonn- MSc Robotics & Geodetic Engineering%
% Alireza.Ahmadi@uni-bonn.de                             %
% AlirezaAhmadi.xyz                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
#include "filter2D.h"

namespace DynaMap{

    filter2D::~filter2D(){}

    __global__ 
    void bilateralFilterKernel(float* dst, float* src, 
                                const float* gaussianKernel, const float e_d, 
                                const int r, rgbdSensor sensor) {

        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        int size = sensor.rows * sensor.cols;
        for (int idx = index; idx < size; idx += stride){

            if(src[idx] == 0){
                dst[idx] = 0;
                continue;
            }

            float sum = 0.0f;
            float t = 0.0f;
            const float center = src[idx];

            for(int u = -r; u <= r; ++u) {
                for(int v = -r; v <= r; ++v) {
                    int index = v * sensor.cols + u + idx;
                    if(index > sensor.cols* sensor.rows)index = sensor.cols* sensor.rows;
                    if(index < 0)index = 0;
                    const float curPix = src[index];
                    if(curPix > 0){
                        const float diff = curPix - center;
                        const float factor = gaussianKernel[abs(u+r)] * gaussianKernel[abs(v+r)] * __expf(-diff*diff / (2 * e_d * e_d));
                        t += factor * curPix;
                        sum += factor;
                    }
                }
            }
            dst[idx] = t / sum;
            if(isnan(dst[idx])){
                dst[idx] = 0;
            }
        }
    }
    void filter2D::bilateralFilter(float* dst, float* src, const gaussianKernal& kernel){
        int threads_per_block = 64;
        int thread_blocks =(sensor.cols * sensor.rows + threads_per_block - 1) / threads_per_block;
        // std::cout << "<<<kernel_bilateralFilter>>> threadBlocks: "<< thread_blocks << ", threadPerBlock: " << threads_per_block << std::endl;
        bilateralFilterKernel <<< thread_blocks, threads_per_block >>>(dst, src, 
                                            kernel.kernel, kernel.filterDelta, 
                                            kernel.kernelRadius, sensor);
        cudaDeviceSynchronize();
        if(cudaGetLastError())std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    }

}  // namespace DynaMap