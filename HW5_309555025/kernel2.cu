#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
__global__ void mandelKernel(int* d_img, int maxIter, float stepX, float stepY, float lowerX, float lowerY) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int thisX = blockDim.x*blockIdx.x+threadIdx.x; //0~1599
    int thisY = blockIdx.y; //0~1199
    int index = (thisY * 1600)  + thisX;
    float x = lowerX + thisX * stepX;
    float y = lowerY + thisY * stepY;
    //
    int i;
    float z_x = x;
    float z_y = y;
    for(i=0;i<maxIter;i++){
	if(z_x*z_x + z_y*z_y > 4.f)
		break;
	float new_x = z_x*z_x - z_y*z_y;
	float new_y = 2.f * z_x * z_y;
        z_x = x + new_x;
	z_y = y + new_y;	
    }
    d_img[index] = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    //
    int width = resX; 
    int height = resY;
    int N = width*height;
    int *d_img;
    size_t pitch;
    //
    // cudaHostAlloc(&h_img,N*sizeof(int),cudaHostAllocMapped);
    // cudaHostGetDevicePointer(&d_img,h_img,0);
    cudaMallocPitch(&d_img, &pitch, width*sizeof(int),height);
    //
    dim3 blockSize(400);
    dim3 blockNum(4,1200);
    mandelKernel<<<blockNum,blockSize>>>(d_img, maxIterations,stepX,stepY,lowerX,lowerY);
    //
    cudaDeviceSynchronize();
    //
    cudaMemcpy(img,d_img,N*sizeof(int),cudaMemcpyDeviceToHost);
    //
    //cudaFreeHost(h_img);
    //cudaFreeHost(h_maxIter);
    //cudaFreeHost(h_dx);
    //cudaFreeHost(h_dy);
    //cudaFreeHost(h_lx);
    //cudaFreeHost(h_ly);
}
