__kernel void convolution(__global float *inputImg, 
			__global float *outputImg,
			__global float *filter,
			const int filterWidth,
			const int imgWidth,
			const int imgHeight
			)
{
	const int j = get_global_id(0);
	const int i = get_global_id(1);
	const int idx = i*imgWidth + j;
	float sum = 0;
	int k,l;
	const int halfFilterSize = filterWidth / 2;
	for(k = -halfFilterSize;k <= halfFilterSize;k++){
		for(l = -halfFilterSize; l <= halfFilterSize;l++){
			if(i+k >= 0 && i+k < imgHeight && j+l >= 0 && j+l < imgWidth){
				sum += inputImg[(i+k)*(imgWidth)+j+l] * filter[(k+halfFilterSize)*filterWidth + l + halfFilterSize];
			}
		}
	}
	outputImg[idx] = (float)sum;	
}
