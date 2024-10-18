#ifndef __sellCSigmaMatrix__
#define __sellCSigmaMatrix__

#include<vector>
#include<stdexcept>
#include<cassert>
#include<iostream>
#include<cmath>
#include"crsMatrix.h"
#include<chrono>

#define AssertCuda(error_code)                                               \
if (error_code != cudaSuccess)                                               \
          {                                                                  \
          std::cout << "The cuda call in " << __FILE__ << " on line "        \
           << __LINE__ << " resulted in the error '"                         \
           << cudaGetErrorString(error_code) << "'" << std::endl;            \
           std::abort();                                                     \
 }



template<typename Number>
 __global__ void sellCSigmagemvKernel(const Number *A, const Number *x, Number *y, const int *row_offset_mapping, const int *row_largest_col_mapping, const int *colIdx,  const unsigned N, const int C){
    
  
    //Global thread index
    unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned i = threadIdx.x;
 
        if ((tid < N) && (i < C)){
         
            Number temp = 0.0;
            for(int j = 0; j < row_largest_col_mapping[tid]; j++){
                // Multiply the matrix value by the corresponding element in the vector x
                temp += A[row_offset_mapping[tid] + i + j*C] * x[colIdx[row_offset_mapping[tid] + i +j*C]];
            }
            y[tid] = temp; 
         }
   
}



template<class Number>
class sellCSigmaMatrix{
    //Class to wrap around the SELL-C-sigma format
    public:
        //Constructor
        sellCSigmaMatrix(typename crsMatrix<Number>::sell_c_sigma_data_t sell_c_sigma_data);

        //Selectors
        int get_C() const {return C;}
        int get_total_elems()const {return total_elems;}
        int get_sigma() const {return sigma;}
        int get_nnz() const {return nnz;}
        int get_n_rows() const {return n_rows;}
        int get_n_cols() const {return n_cols;}
        const int * get_largestnnz_ptr() const {return &largestnnz[0];}
        const int * get_offset_ptr() const {return &offset_ptr[0];}
        const int * get_col_ptr() const {return &col_idx[0];}
        const Number * get_static_values_ptr() const {return &values[0];}
        const int *get_row_offset_mapping_ptr() const {return &row_offset_mapping[0];}
        const int *get_row_largest_col_mapping_ptr() const {return &row_largest_col_mapping[0];}

        //Methods
        static void gemv(const sellCSigmaMatrix<Number> &A, const std::vector<Number> &x, std::vector<Number> &y);
        static void cugemv(const sellCSigmaMatrix<Number> &A, const std::vector<Number> &x, std::vector<Number> &y);

    private:
        int C;
        int sigma;
        int nnz; //Non zero elements
        int n_rows; 
        int n_cols;
        int total_elems;
        std::vector<int> largestnnz;
        std::vector<int> offset_ptr; //Offset pointers
        std::vector<int> col_idx; //Column index
        std::vector<Number> values; //Values, with padded 0s
        std::vector<int> row_largest_col_mapping;
        std::vector<int> row_offset_mapping;

};

template<class Number>
sellCSigmaMatrix<Number>::sellCSigmaMatrix(typename crsMatrix<Number>::sell_c_sigma_data_t sell_c_sigma_data){
    C = sell_c_sigma_data.C;
    sigma = sell_c_sigma_data.sigma;
    nnz = sell_c_sigma_data.nnz;   
    n_rows = sell_c_sigma_data.n_rows; 
    n_cols = sell_c_sigma_data.n_cols;
    total_elems = sell_c_sigma_data.total_elems;
    largestnnz = std::move(sell_c_sigma_data.largestnnz);
    offset_ptr = std::move(sell_c_sigma_data.offset_ptr); 
    col_idx = std::move(sell_c_sigma_data.col_idx); 
    values = std::move(sell_c_sigma_data.values); 
    row_largest_col_mapping = std::move(sell_c_sigma_data.row_largest_col_mapping);
    row_offset_mapping = std::move(sell_c_sigma_data.row_offset_mapping);



}
template<class Number>
void sellCSigmaMatrix<Number>::gemv(const sellCSigmaMatrix<Number> &A, const std::vector<Number> &x, std::vector<Number> &y){
    int n_rows = A.get_n_rows();
    int n_cols = A.get_n_cols();

    if (x.size() != n_cols || y.size() != n_rows) {
        throw std::invalid_argument("Vector dimensions do not match matrix dimensions");
    }

    const int *largestnnz_ptr = A.get_largestnnz_ptr();
    const int *offset_ptr = A.get_offset_ptr();
    const int *colIdx = A.get_col_ptr();
    const Number *values = A.get_static_values_ptr();
    const int C = A.get_C();

    for(int i = 0; i<n_rows;i++){
        Number temp = 0.0;
        const int block_num = i / C;
        for(int j = 0; j < largestnnz_ptr[block_num]; j++){
           // Compute the index into values and colIdx
            int index = offset_ptr[block_num] + j * C + i % C;

            // Multiply the matrix value by the corresponding element in the vector x
            temp += values[index] * x[colIdx[index]];
        }
        y[i] = temp; 
    }

}


template<class Number>
void sellCSigmaMatrix<Number>::cugemv(const sellCSigmaMatrix<Number> &A, const std::vector<Number> &x, std::vector<Number> &y){
    int n_rows = A.get_n_rows();
    int n_cols = A.get_n_cols();

    if (x.size() != n_cols || y.size() != n_rows) {
        throw std::invalid_argument("Vector dimensions do not match matrix dimensions");
    }

    const int *largestnnz_ptr = A.get_largestnnz_ptr();
    const int *offset_ptr = A.get_offset_ptr();
    const int * col_ptr = A.get_col_ptr();
    const Number *values = A.get_static_values_ptr();
    const int nnz = A.get_nnz();
    const int total_elems = A.get_total_elems();   
    const int C = A.get_C();
    const int *row_offset_mapping_ptr = A.get_row_offset_mapping_ptr();
    const int *row_largest_col_mapping_ptr = A.get_row_largest_col_mapping_ptr();
    
    
    //For kernel call
    const int block_size = 128;
    const unsigned int n_blocks = (n_rows + block_size - 1) / block_size;
    

    //Device ptrs
    Number *A_device;
    int *offset_ptr_device;
    int *colIdx_device;
    int *largestnnz_device; 
    int *row_largest_col_mapping_device;
    int *row_offset_mapping_device;

    int n_blocks_sell_c_sigma = (int) ceil((double)n_rows / (double)C);
    printf("n_blocks = %d\n", n_blocks_sell_c_sigma);

    Number *y_device;
    Number *x_device;


    //Allocate on device
    AssertCuda(cudaMalloc(&A_device, total_elems * sizeof(Number)));
    AssertCuda(cudaMalloc(&offset_ptr_device, (n_blocks_sell_c_sigma+1) * sizeof(int)));
    AssertCuda(cudaMalloc(&colIdx_device, total_elems*sizeof(int)));
    AssertCuda(cudaMalloc(&largestnnz_device, n_blocks_sell_c_sigma*sizeof(int))); 
    AssertCuda(cudaMalloc(&x_device, x.size() * sizeof(Number)));
    AssertCuda(cudaMalloc(&y_device, y.size() * sizeof(Number)));
    AssertCuda(cudaMalloc(&row_largest_col_mapping_device, n_rows*sizeof(int))); 
    AssertCuda(cudaMalloc(&row_offset_mapping_device, n_rows*sizeof(int))); 
    
    //Copy
    AssertCuda(cudaMemcpy(&A_device[0], &values[0], total_elems*sizeof(Number), cudaMemcpyHostToDevice));
    AssertCuda(cudaMemcpy(&offset_ptr_device[0], &offset_ptr[0], (n_blocks_sell_c_sigma+1)*sizeof(int), cudaMemcpyHostToDevice));
    AssertCuda(cudaMemcpy(&colIdx_device[0], &col_ptr[0], total_elems*sizeof(int), cudaMemcpyHostToDevice));
    AssertCuda(cudaMemcpy(&largestnnz_device[0], &largestnnz_ptr[0], n_blocks_sell_c_sigma*sizeof(int), cudaMemcpyHostToDevice));
    AssertCuda(cudaMemcpy(&x_device[0], &x[0], x.size()*sizeof(Number), cudaMemcpyHostToDevice));
    AssertCuda(cudaMemcpy(&y_device[0], &y[0], y.size()*sizeof(Number), cudaMemcpyHostToDevice));
    AssertCuda(cudaMemcpy(&row_largest_col_mapping_device[0], &row_largest_col_mapping_ptr[0], n_rows*sizeof(Number), cudaMemcpyHostToDevice));
    AssertCuda(cudaMemcpy(&row_offset_mapping_device[0], &row_offset_mapping_ptr[0], n_rows*sizeof(Number), cudaMemcpyHostToDevice));
   

    cudaDeviceSynchronize();
    const auto t1 = std::chrono::steady_clock::now();
    sellCSigmagemvKernel<Number><<<n_blocks, block_size>>>(&A_device[0], &x_device[0], &y_device[0], &row_offset_mapping_device[0], &row_largest_col_mapping_device[0],  &colIdx_device[0], n_rows, C);
    cudaDeviceSynchronize();
    const float time = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::steady_clock::now() - t1).count();
    printf("Took %.17lf to compute square Ax = y of size N=%d on GPU\n", time, n_rows);
    Number BW = sizeof(Number) * total_elems + 4*total_elems + 2*(n_rows) * 4 + 2*n_rows * sizeof(Number);
    BW *=  (1e-9)/time;
    printf("BW = %lf\n", BW);
    printf("Total elems in sparse matrix: %d\n", total_elems);
    //Copy back to device
    AssertCuda(cudaMemcpy(&y[0], &y_device[0], y.size() * sizeof(Number), cudaMemcpyDeviceToHost));
    
    
    cudaDeviceSynchronize();
    //Free on device
    cudaFree(A_device);
    cudaFree(offset_ptr_device);
    cudaFree(colIdx_device);
    cudaFree(largestnnz_device);
    cudaFree(y_device);
    cudaFree(x_device);
    cudaFree(row_largest_col_mapping_device);
    cudaFree(row_offset_mapping_device);

}    
#endif
