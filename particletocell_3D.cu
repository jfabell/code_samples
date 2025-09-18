#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// For each grid cell, reduce over subcells
__global__ void reduction_cells(double *w, int r, int RM_total, int M_total)
{
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    int j;
    int tid = threadIdx.x;

    // If valid grid cell index
    if(i < M_total) {

        // create memory to store temp sums and load elements
        extern __shared__ double temp_sum[];

        // move the w values into tempsum for this thread
        for (j = 0; j < r; j++) {
            temp_sum[r*tid + j] = w[r*i + j];
        }
        __syncthreads();
   
        // for each grid cell, do the sum, putting the value ultimately into the "0" idx
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {

            if (tid < s) {
                for(j = 0; j < r; j++) {
                    temp_sum[r * tid + j] += temp_sum[r * (tid + s) + j];
                }
            }

            // the values for r for this cell are now fully summed
            __syncthreads();
        }

        // each block writes uses thread 0 to write
        // results back into w from temp_sum
        if (tid == 0){

            for(j = 0; j < r; j++) {
                w[r*blockIdx.x+j] = temp_sum[j];
            }

        }

    }
}


// Does the atomic reductions for each subcell
__global__ void reduction_subcells(double *w, double *v, double *x0, double *x1,
                                   double *x2, int r, int Nc, int M0, int M1,
                                   int M2, int M_total) 
{

    int i=threadIdx.x + (blockIdx.x * blockDim.x);
    int j=threadIdx.y + (blockIdx.y * blockDim.y);

    // if valid particle and valid data point indexes
    if(i < Nc && j < r) {

        // Grab 3D bin of particle i mapped to 1D
        int b = M1*M2*int(x0[i]*float(M0)) + M2*int(x1[i]*float(M1)) + int(x2[i]*float(M2));

        // Atomically add in the value j for particle i
        atomicAdd(&w[r*b+j] , v[r*i+j]);

    }

}


// Main function called by cuda_red3D

/*
Performs 3D particle-to-cell summation for a set of 
Nc particles with weights dw_ar and 3D positions dxD_ar
and r different summation values dv_ar

-- Nc: particle count 
-- r: num data points per particle
-- MD: # grid cells in dimension D
-- RMD: # total subcells in dimension D
-- dxD_ar: dimension D normalized positions for particles (len Nc each)
-- dw_ar: 1D weight arrary for particles (len Nc)
-- dv_ar: 1D array of particle data points (len r*Nc)

*/
void calls(int Nc, int r,
           int M0, int M1, int M2,
           int RM0, int RM1, int RM2,
           py::array_t<double>& dw_ar, 
           py::array_t<double>& dv_ar, 
           py::array_t<double>& dx0_ar, 
           py::array_t<double>& dx1_ar, 
           py::array_t<double>& dx2_ar)
{

    // Total grid sizes
    int M_total = M0*M1*M2;
    int RM_total = RM0*RM1*RM2;

    // Setup for 1st kernel //
    int blockx = 256; // split up particles into chunks of 256
    int blocky = r; // to match num. values
    dim3 tpb(blockx, blocky, 1);
    dim3 bpg(ceil(Nc/blockx) + 1, ceil(r / blocky) + 1, 1);

    // Setup for 2nd kernel // 
    // num. values per cell = num. values * subells per cell
    int bloks = ceil((double) M_total/RM_total ); 
    size_t ns = bloks * r * sizeof(double); // memory for temp arrays
    dim3 tpb2(bloks, 1, 1);
    dim3 bpg2(ceil(M_total / bloks) + 1, 1, 1);

    // Recasting the arrays from Python
    double* dw  = static_cast<double *>(dw_ar.request().ptr);
    double* dv  = static_cast<double *>(dv_ar.request().ptr);
    double* dx0 = static_cast<double *>(dx0_ar.request().ptr);
    double* dx1 = static_cast<double *>(dx1_ar.request().ptr);
    double* dx2 = static_cast<double *>(dx2_ar.request().ptr);

    // Call the subcell reductions
    reduction_subcells<<<bpg, tpb>>>(dw, dv, dx0, dx1, dx2, r, Nc, M0, M1, M2, M_total);

    // If multiple subcells used per cell, do local summations
    if ((M_total / RM_total) > 1){
        reduction_cells<<<bpg2, tpb2, ns>>>(dw, r, RM_total, M_total);
    }

}

PYBIND11_MODULE(cuda_red3D, m) {
    m.def("calls", &calls);
}



