/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% by: Alireza Ahmadi                                     %
% University of Bonn- MSc Robotics & Geodetic Engineering%
% Alireza.Ahmadi@uni-bonn.de                             %
% AlirezaAhmadi.xyz                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
#include "pyramid.h"

namespace DynaMap{

    pyramid::~pyramid(){}

    // __global__
    // void upSampleKernel(float *src, float *dst){
    // }
    __global__
    void downSampleKernel(float *src, float *dst, rgbdSensor sensor,int scale){

        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        int size = sensor.cols * sensor.rows;
        
        for (int idx = index; idx < size; idx += stride){
            int v = static_cast<int>(idx / sensor.cols);
            int u = static_cast<int>(idx - sensor.cols * v);

            int u_offset = 0;
            int v_offset = 0;
            if(scale == 2){
                u_offset = 160;
                v_offset = 120;
            }else if(scale == 4){
                u_offset = 240;
                v_offset = 180;
            }else if(scale == 8){
                u_offset = 280;
                v_offset = 210;
            }else{
                u_offset = 0;
                v_offset = 0;
            }

            if( v >= 0 && v < sensor.rows && 
                u >= 0 && u < sensor.cols ) {
                if( v % scale != 0 && u % scale != 0){
                    uint py_u = __float2uint_rd(u / scale) + u_offset;
                    uint py_v = __float2uint_rd(v / scale) + v_offset;
                    dst[py_v * sensor.cols + py_u] = src[v * sensor.cols + u];
                }
            }
        }
    }
    void pyramid::downPyramid(float *src, int scale){
        if(scale != 1){
            cudaMallocManaged(&dstBlur, sizeof(float) * sensor.rows * sensor.cols);
            cudaDeviceSynchronize();

            float gaussKernel[25] = {1.0,4.0,6.0,4.0,1.0,
                                    4.0,16.0,24.0,16.0,4.0,
                                    6.0,24.0,36.0,24.0,6.0,
                                    4.0,16.0,24.0,16.0,4.0,
                                    1.0,4.0,6.0,4.0,1.0};
            // Convolving guassian kenrnel on image
            convolve(src, dstBlur, gaussKernel);    // applying gaussian blur
            // removing even columns and rows in parallel from the source image
            int threads_per_block = 64;
            int thread_blocks =(sensor.cols * sensor.rows + 
                threads_per_block - 1) / threads_per_block;
            // std::cout << "<<<kernel_downSampleKernel>>> threadBlocks: "<< thread_blocks << ", threadPerBlock: " << threads_per_block << std::endl;
            downSampleKernel<<<thread_blocks, threads_per_block>>>(src, depth, sensor, scale);
            cudaDeviceSynchronize();
            if(cudaGetLastError())std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
            
            // cudaDeviceSynchronize();
            // cudaFree(dstBlur);
        }else{
            depth = src;
        }
    }
    __global__   // todo ...
    void replicationPaddingKernel(){
    }

}  // namespace DynaMap



// for (int idx = 0; idx < 100; idx++){
//     int v = static_cast<int>(idx / 10);
//     int u = static_cast<int>(idx - 10 * v);
    
//     if( v >= 0 && v < 10 && 
//         u >= 0 && u < 10 ) {
//             std:: cout << "heree .." << std::endl;
//         if( v % 2 != 0 && u % 2 != 0){
//             uint py_u = floor(u / 2);
//             uint py_v = floor(v / 2);
//             std:: cout << "pyidx: "<< py_u * 5 + py_v  << ", origidx: " << u * 10 + v<< std::endl;
//         }
//     }
// }
