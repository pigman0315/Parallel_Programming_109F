#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,

            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;
    int imgSize = imageWidth * imageHeight;
    //
    cl_context gpu_context = *context;
    cl_device_id gpu_device = *device;
    cl_program gpu_program = *program;
    // create queue
    cl_command_queue queue;
    queue = clCreateCommandQueue(gpu_context,gpu_device,0,&status);
    // args: filter, img, fliterWidth, imgHeight, imgWidth
    cl_mem inputImg_cl = clCreateBuffer(gpu_context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(float)*imgSize,inputImage, NULL);
    cl_mem filter_cl = clCreateBuffer(gpu_context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(float)*filterSize,filter,NULL);
    cl_mem outputImg_cl = clCreateBuffer(gpu_context, CL_MEM_WRITE_ONLY, sizeof(float)*imgSize, outputImage, NULL);
    // create kernel
    cl_kernel conv = clCreateKernel(gpu_program, "convolution",0);
    // set arguments
    clSetKernelArg(conv,0,sizeof(cl_mem),(void *)&inputImg_cl);
    clSetKernelArg(conv,1,sizeof(cl_mem),(void *)&outputImg_cl);
    clSetKernelArg(conv,2,sizeof(cl_mem),(void *)&filter_cl);
    clSetKernelArg(conv,3,sizeof(int),(void *)&filterWidth);
    clSetKernelArg(conv,4,sizeof(int),(void *)&imageWidth);
    clSetKernelArg(conv,5,sizeof(int),(void *)&imageHeight);
    //
    const int workDimension = 2;
    size_t local_work_size[2];
    local_work_size[0] = 50;
    local_work_size[1] = 20;
    size_t global_work_size[2];
    global_work_size[0] = imageWidth;
    global_work_size[1] = imageHeight;
    cl_event evt, r_evt;
    // execute kernel
    status = clEnqueueNDRangeKernel(queue, conv, 2,NULL,global_work_size,local_work_size,0,NULL,NULL);
    // status = clEnqueueNDRangeKernel(queue, conv, workDimension,NULL,global_work_size,NULL,0,NULL,NULL);
    if(status == CL_SUCCESS){
	status = clEnqueueReadBuffer(queue, outputImg_cl, CL_TRUE, 0, sizeof(float)*imgSize, outputImage,0,NULL,NULL);
    }
}
