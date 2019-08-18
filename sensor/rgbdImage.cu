/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% by: Alireza Ahmadi                                     %
% University of Bonn- MSc Robotics & Geodetic Engineering%
% Alireza.Ahmadi@uni-bonn.de                             %
% AlirezaAhmadi.xyz                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
#include "sensor/rgbdImage.h"

namespace DynaMap{

    rgbdImage::~rgbdImage(){
        cudaDeviceSynchronize();
        cudaFree(rgb);
        cudaFree(depth);
    }
    void rgbdImage::init(const rgbdSensor& _sensor){
        sensor = _sensor;
        cudaMallocManaged(&rgb, sizeof(uchar3) * sensor.rows * sensor.cols);
        cudaMallocManaged(&depth, sizeof(float) * sensor.rows * sensor.cols);
        cudaDeviceSynchronize();
    }
    __global__ 
    void creatPointCloudKernel(float* depth, rgbdSensor sensor, 
                                             float *pointsX,
                                             float *pointsY,
                                             float *pointsZ,
                                             int scale) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        int size = sensor.rows * sensor.cols;
        geometry::PointXYZ point;
        for (int idx = index; idx < size; idx += stride){
            if(depth[idx] <= 0.0) continue;
            point = GetPoint3d(idx, depth[idx], sensor);
            pointsX[idx] = point.x;
            pointsY[idx] = point.y;
            pointsZ[idx] = point.z;
        }
    }
    void rgbdImage::getPointCloudXYZ(geometry::PointCloudXYZ& cloud, int scale){
        int threads_per_block = 256;
        int thread_blocks =(sensor.cols * sensor.rows + threads_per_block - 1) / threads_per_block;
        //std::cout << "<<<kernel_creatPointCloud>>> threadBlocks: "<< thread_blocks << ", threadPerBlock: " << threads_per_block << std::endl;
        creatPointCloudKernel <<< thread_blocks, threads_per_block >>> (this->depth, 
                                                                        this->sensor, 
                                                                        cloud.x, 
                                                                        cloud.y, 
                                                                        cloud.z,
                                                                        scale);
        cudaDeviceSynchronize();
        if(cudaGetLastError())std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    }
    __global__
    void depthToNormalKernel(float* depth, rgbdSensor sensor, 
                                            float *normalsX, 
                                            float *normalsY, 
                                            float *normalsZ){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        int size = sensor.rows * sensor.cols;
        for (int idx = index; idx < size; idx += stride) {
            // int v = static_cast<int>(idx / sensor.cols);
            // int u = static_cast<int>(idx - sensor.cols * v);
            int step = 1;

            int index = idx + step;
            int right = (index >= size ) ? idx : index;
            index = idx - step;
            int left = (index <= 0 ) ? idx : index;
            
            index = idx + sensor.cols * step;
            int down = (index >= size ) ? idx : index;
            index = idx - sensor.cols * step;
            int up = (index <= 0 ) ? idx : index;

            if(depth[right] == 0 || depth[left] == 0 || depth[down] == 0 || depth[up] == 0)continue;  
            // todo should it set to zero ???

            float dzdx = (depth[right]*sensor.depthScaler - depth[left]*sensor.depthScaler) / 2.0;
            float dzdy = (depth[down]*sensor.depthScaler  - depth[up]*sensor.depthScaler)   / 2.0;

            float3 direction = make_float3(dzdx, dzdy, 1.0f);
            float3 n = normalize(direction);
            normalsX[idx]  = n.x;
            normalsY[idx]  = n.y;
            normalsZ[idx]  = n.z;
        }
    }
    void rgbdImage::getNormalsfromDepthImage(geometry::NormalsXYZ& normals){
        int threads_per_block = 256;
        int thread_blocks =(sensor.cols * sensor.rows + threads_per_block - 1) / threads_per_block;
        // std::cout << "<<<kernel_depthToNormal>>> threadBlocks: "<< thread_blocks << ", threadPerBlock: " << threads_per_block << std::endl;
        depthToNormalKernel <<< thread_blocks, threads_per_block >>> (depth, 
                                                                       sensor,
                                                                       normals.x, 
                                                                       normals.y, 
                                                                       normals.z);
        cudaDeviceSynchronize();
        if(cudaGetLastError())std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
        
    }
    cv::Mat rgbdImage::testNormalsfromDepthImage(cv::Mat& depth, float depthScalar){
        geometry::NormalsXYZ Normals;
        depth.convertTo(depth, CV_32FC1, depthScalar);
        cv::Mat NormalImage(sensor.rows, sensor.cols, CV_32FC3);
        cudaMallocManaged(&Normals.x, sizeof(float) * sensor.rows * sensor.cols);
        cudaMallocManaged(&Normals.y, sizeof(float) * sensor.rows * sensor.cols);
        cudaMallocManaged(&Normals.z, sizeof(float) * sensor.rows * sensor.cols);

        for (uint i = 0; i < sensor.rows; i++) {
            for (uint j = 0; j < sensor.cols; j++) {
                    this->depth[i * sensor.cols + j] = depth.at<float>(i, j);
            }
        }

        this->getNormalsfromDepthImage(Normals);

        for (int i = 0; i < sensor.rows; i++) {
            for (int j = 0; j < sensor.cols; j++) {
                NormalImage.at<cv::Vec3f>(i, j)[0] = Normals.x[i * sensor.cols + j];
				NormalImage.at<cv::Vec3f>(i, j)[1] = Normals.y[i * sensor.cols + j];
				NormalImage.at<cv::Vec3f>(i, j)[2] = Normals.z[i * sensor.cols + j];
            }
        }
		cudaFree(Normals.x);
        cudaFree(Normals.y);
        cudaFree(Normals.z);
        return NormalImage;
    }
    __global__ 
    void vertexToNormalKernel(rgbdSensor sensor, float *normalsX, 
                                                 float *normalsY, 
                                                 float *normalsZ, 
                                                 float *pointsX,
                                                 float *pointsY,
                                                 float *pointsZ){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        int size = sensor.cols * sensor.rows;

        for (int idx = index; idx < size; idx += stride) {
            // int v = static_cast<int>(idx / sensor.cols);
            // int u = static_cast<int>(idx - sensor.cols * v);
            int step = 1;

            int index = idx - step;
            int left_idx = (index < 0 ) ? idx : index;
            const float3 left = make_float3(pointsX[left_idx], pointsY[left_idx], pointsZ[left_idx]);
            index = idx + step;
            int right_idx = (index >= size ) ? idx : index;
            const float3 right = make_float3(pointsX[right_idx], pointsY[right_idx], pointsZ[right_idx]);
            index = idx - sensor.cols;
            int up_idx = (index <= 0 ) ? idx : index;
            const float3 up = make_float3(pointsX[up_idx], pointsY[up_idx], pointsZ[up_idx]);
            index = idx + sensor.cols;
            int down_idx = (index >= size ) ? idx : index;
            const float3 down = make_float3(pointsX[down_idx], pointsY[down_idx], pointsZ[down_idx]);

            if(left.z == 0 || right.z == 0 || up.z == 0 || down.z == 0)continue;

            const float3 dzdx = right - left;
            const float3 dzdy = down - up;
            float3 normal = normalize(cross(dzdy, dzdx)); // switched dx and dy to get factor -1
            normalsX[idx]  = normal.x;
            normalsY[idx]  = normal.y;
            normalsZ[idx]  = normal.z;
        }
    }
    void rgbdImage::getNormalsfromVertices(geometry::NormalsXYZ& normals, geometry::PointCloudXYZ cloud){
        int threads_per_block = 256;
        int thread_blocks =(sensor.cols * sensor.rows + threads_per_block - 1) / threads_per_block;
        //std::cout << "<<<kernel_vertexToNormal>>> threadBlocks: "<< thread_blocks << ", threadPerBlock: " << threads_per_block << std::endl;
        vertexToNormalKernel <<< thread_blocks, threads_per_block >>> (sensor, normals.x,
                                                                                 normals.y,
                                                                                 normals.z,
                                                                                 cloud.x,
                                                                                 cloud.y,
                                                                                 cloud.z);
        cudaDeviceSynchronize();
        if(cudaGetLastError())std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    }
    cv::Mat rgbdImage::testNormalsfromVertices(cv::Mat& depth, float depthScalar){
        geometry::NormalsXYZ Normals;
        geometry::PointCloudXYZ testPCL;
        depth.convertTo(depth, CV_32FC1, depthScalar);
        cv::Mat NormalImage(sensor.rows, sensor.cols, CV_32FC3);
        cudaMallocManaged(&Normals.x, sizeof(float) * sensor.rows * sensor.cols);
        cudaMallocManaged(&Normals.y, sizeof(float) * sensor.rows * sensor.cols);
        cudaMallocManaged(&Normals.z, sizeof(float) * sensor.rows * sensor.cols);

        cudaMallocManaged(&testPCL.x, sizeof(float) * sensor.rows * sensor.cols);
        cudaMallocManaged(&testPCL.y, sizeof(float) * sensor.rows * sensor.cols);
        cudaMallocManaged(&testPCL.z, sizeof(float) * sensor.rows * sensor.cols);

        for (uint i = 0; i < sensor.rows; i++) {
            for (uint j = 0; j < sensor.cols; j++) {
                    this->depth[i * sensor.cols + j] = depth.at<float>(i, j);
            }
        }

        this->getPointCloudXYZ(testPCL);
        this->getNormalsfromVertices(Normals, testPCL);
        // for (size_t i = 1000; i < 1050 ; i++){
        //     std::cout << Normals.x[i] << ", " << Normals.y[i] << ", " << Normals.z[i] << std::endl;
        // }
        for (int i = 0; i < sensor.rows; i++) {
            for (int j = 0; j < sensor.cols; j++) {
                NormalImage.at<cv::Vec3f>(i, j)[0] = static_cast<float>(Normals.x[i * sensor.cols + j]);
				NormalImage.at<cv::Vec3f>(i, j)[1] = static_cast<float>(Normals.y[i * sensor.cols + j]);
				NormalImage.at<cv::Vec3f>(i, j)[2] = static_cast<float>(Normals.z[i * sensor.cols + j]);
            }
        }
		cudaFree(Normals.x);
        cudaFree(Normals.y);
        cudaFree(Normals.z);

        cudaFree(testPCL.x);
        cudaFree(testPCL.y);
        cudaFree(testPCL.z);
        return NormalImage;
    }

    cv::Mat rgbdImage::testNormalsfromDepthImageCV(cv::Mat& depth, float depthScalar){
        depth.convertTo(depth, CV_32FC1, depthScalar);  // check depth scalar
        cv::Mat CVnormals(depth.size(), CV_32FC3);
		for(int x = 0; x < depth.rows; ++x){
			for(int y = 0; y < depth.cols; ++y){
				float dzdx = (depth.at<float>(x+1, y) - depth.at<float>(x-1, y)) / 2.0;
				float dzdy = (depth.at<float>(x, y+1) - depth.at<float>(x, y-1)) / 2.0;

				cv::Vec3f d(-dzdx, -dzdy, 1.0f);

				cv::Vec3f n = cv::normalize(d);
				CVnormals.at<cv::Vec3f>(x, y) = n;
			}
		}
		return CVnormals;
    }

    cv::Mat rgbdImage::getCVImagefromCudaDepth(void){
        cv::Mat _depth(sensor.rows, sensor.cols, CV_32FC1);
        for (int i = 0; i < sensor.rows; i++) {
            for (int j = 0; j < sensor.cols; j++) {
                _depth.at<float>(i, j) = static_cast<float>(this->depth[i * sensor.cols + j]);
            }
        }
        return _depth;
    }
    void rgbdImage::getCudafromCVDepth(cv::Mat cvDepth){
        for (uint i = 0; i < sensor.rows; i++) {
            for (uint j = 0; j < sensor.cols; j++) {
                this->depth[i * sensor.cols + j] = cvDepth.at<float>(i, j);
            }
        }
    }
}  // namespace DynaMap




