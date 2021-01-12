#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
__global__ void mandelKernel(int* d_img, const int maxIter, const float stepX, const float stepY, const float lowerX, const float lowerY) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    // int thisX = blockDim.x*blockIdx.x+threadIdx.x; //0~1599
    // int thisY = blockIdx.y; //0~1199
    int horz_idx = blockIdx.x % 50;
    int vert_idx = blockIdx.x / 50;
    int thisX = horz_idx*32 + threadIdx.x;
    int thisY = vert_idx*15 + threadIdx.y;
    const int index = (thisY * 1600)  + thisX;
    const float x = lowerX + thisX * stepX;
    const float y = lowerY + thisY * stepY;
    //
    int i;
    float z_x = x;
    float z_y = y;
    for(i=0;i<maxIter;i++){
	if(z_x*z_x + z_y*z_y > 4.f)
		break;
	const float new_x = z_x*z_x - z_y*z_y;
	const float new_y = 2.f * z_x * z_y;
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
    int *d_img, *h_img;
    size_t pitch;
    //
    //cudaHostAlloc(&h_img,N*sizeof(int),cudaHostAllocMapped);
    //cudaHostGetDevicePointer(&d_img,h_img,0);
    cudaMallocPitch(&d_img, &pitch, width*sizeof(int),height);
    //
    dim3 blockSize(32,15);
    dim3 blockNum(4000);
    mandelKernel<<<blockNum,blockSize>>>(d_img, maxIterations,stepX,stepY,lowerX,lowerY);
    //
    cudaDeviceSynchronize();
    //
    cudaMemcpy(img,d_img,N*sizeof(int),cudaMemcpyDeviceToHost);
    //
}
