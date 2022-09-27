#ifndef KMEANS_CUDA_KERNEL_CUH
#define KMEANS_CUDA_KERNEL_CUH

static texture<float, 1, cudaReadModeElementType> t_features;
// t_features_flipped has the layout point0[dim 0-n-1]point1[dim 0-n-1]
static texture<float, 1, cudaReadModeElementType> t_features_flipped;
static texture<float, 1, cudaReadModeElementType> t_clusters;

__global__ void invert_mapping(float *input,			/* original */
    float *output,			/* inverted */
    int npoints,				/* npoints */
    int nfeatures);			/* nfeatures */

__global__ void kmeansPoint(float  *features,			/* in: [npoints*nfeatures] */
    int     nfeatures,
    int     npoints,
    int     nclusters,
    int    *membership,
    float  *clusters,
    float  *block_clusters,
    int    *block_deltas);
#endif
