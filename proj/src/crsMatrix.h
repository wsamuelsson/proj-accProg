#ifndef __crsMatrix__
#define __crsMatrix__

#include<vector>
#include<stdexcept>
#include<cassert>
#include<iostream>
#include<cmath>
#include<cuda_runtime.h>

#include"matrix.h"




#define AssertCuda(error_code)                                               \
if (error_code != cudaSuccess)                                               \
          {                                                                  \
          std::cout << "The cuda call in " << __FILE__ << " on line "        \
           << __LINE__ << " resulted in the error '"                         \
           << cudaGetErrorString(error_code) << "'" << std::endl;            \
           std::abort();                                                     \
 }










template<typename Number>
 __global__ void crsgemvKernel(const Number *A, const Number *x, Number *y, const int *row_ptr, const int *col_idx, const unsigned N){
    
  
    //Global thread index
    unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
    
 
        if (tid < N){
            Number tmp = 0.0;
            for(unsigned col = row_ptr[tid]; col < row_ptr[tid + 1]; col++)
                tmp += A[col]*x[col_idx[col]];
            y[tid] =   tmp;
    
        }
   
}


 




template<typename Number>
class crsMatrix{
    //Class to wrap around CRS format
    
    
    public:
        //Structure for sell-c-sigma data 
        typedef struct{
            int C;
            int sigma;
            int nnz; //Non zero elements
            int n_rows; 
            int n_cols;
            int total_elems;
            std::vector<int> largestnnz; //store longest row in each block
            std::vector<int> offset_ptr; //Row pointers
            std::vector<int> col_idx; //Column index
            std::vector<Number> values; //Non-zero values

        } sell_c_sigma_data_t;
        //Constructors
        crsMatrix(typename Matrix<Number>::csr_data_t csr_data);

        //Selectors
        int get_n_rows() const {return n_rows;};
        int get_n_cols() const {return n_cols;};
        int get_nnz() const {return nnz;};
        const int * get_row_ptr() const {return &row_ptr[0];};
        const int * get_col_idx__ptr() const {return &col_idx[0];};
        const Number * get_static_values_ptr() const {return &values[0];};

        //Methods
        void print_csr();
        static void gemv(const crsMatrix<Number> &A, const std::vector<Number> &x, std::vector<Number> &y);
        static void cugemv(const crsMatrix<Number> &A, const std::vector<Number> &x, std::vector<Number> &y);
        sell_c_sigma_data_t get_sell_c_sigma(const int C, const int sigma);

    private:
        int nnz;
        int n_rows;
        int n_cols;
        std::vector<int> row_ptr;
        std::vector<int> col_idx;
        std::vector<Number> values;

};


template<typename Number>
crsMatrix<Number>::crsMatrix(typename Matrix<Number>::csr_data_t csr_data){
    nnz = std::move(csr_data.nnz);
    n_rows = std::move(csr_data.n_rows);
    n_cols = std::move(csr_data.n_cols);
    row_ptr = std::move(csr_data.row_ptr);
    col_idx = std::move(csr_data.col_idx);
    values = std::move(csr_data.values);

}

template<typename Number>
void crsMatrix<Number>::print_csr() {
    
    // Loop through each row
    for (int i = 0; i < n_rows; ++i) {
        // For each row, get the start and end indices from row_ptr
        int start_index = row_ptr[i];         // Starting index for the row
        int end_index = row_ptr[i + 1];      // Ending index for the row

        // Loop through the non-zero entries for the current row
        for (int j = start_index; j < end_index; ++j) {
            int col_index = col_idx[j];      // Get the column index
            Number value = values[j];        // Get the corresponding value

            //MATLAB style printing; one based index
            std::cout << "(" << i+1 << ", " << col_index+1 << ") " << value << std::endl;
        }
    }
}

template<typename Number>
void crsMatrix<Number>::gemv(const crsMatrix<Number> &A, const std::vector<Number> &x, std::vector<Number> &y){
    int n_rows = A.get_n_rows();
    int n_cols = A.get_n_cols();

    if (x.size() != n_cols || y.size() != n_rows) {
        throw std::invalid_argument("Vector dimensions do not match matrix dimensions");
    }

    const int *row_ptr = A.get_row_ptr();
    const int *colIdx = A.get_col_idx__ptr();
    const Number *values = A.get_static_values_ptr();


    for(int i = 0; i < n_rows; i++){
        int row_start = row_ptr[i];
        int row_end = row_ptr[i+1];
        Number temp = 0.0;
        for(int j = row_start; j < row_end; j++){
            temp += values[j] * x[colIdx[j]];
        }
        y[i] = temp;
    }
}

template<typename Number>
void crsMatrix<Number>::cugemv(const crsMatrix<Number> &A, const std::vector<Number> &x, std::vector<Number> &y){
    int n_rows = A.get_n_rows();
    int n_cols = A.get_n_cols();

    if (x.size() != n_cols || y.size() != n_rows) {
        throw std::invalid_argument("Vector dimensions do not match matrix dimensions");
    }

    const int *row_ptr = A.get_row_ptr();
    const int *colIdx = A.get_col_idx__ptr();
    const Number *values = A.get_static_values_ptr();
    const int nnz = A.get_nnz();
    
    //For kernel call
    const int block_size = 128;
    const unsigned int n_blocks = (n_rows + block_size - 1) / block_size;
 

    //Device ptrs
    Number *A_device;
    int *row_ptr_device;
    int *colIdx_device;
    Number *x_device;
    Number *y_device;

    //Allocate on device
    AssertCuda(cudaMalloc(&A_device, nnz * sizeof(Number)));
    AssertCuda(cudaMalloc(&row_ptr_device, (n_rows+1) * sizeof(int)));
    AssertCuda(cudaMalloc(&colIdx_device, nnz*sizeof(int)));
    AssertCuda(cudaMalloc(&y_device, y.size() *sizeof(Number)));
    AssertCuda(cudaMalloc(&x_device, x.size() *sizeof(Number)));
      
    //Copy
    AssertCuda(cudaMemcpy(&A_device[0], &values[0], nnz*sizeof(Number), cudaMemcpyHostToDevice));
    AssertCuda(cudaMemcpy(&row_ptr_device[0], &row_ptr[0], (n_rows+1)*sizeof(int), cudaMemcpyHostToDevice));
    AssertCuda(cudaMemcpy(&colIdx_device[0], &colIdx[0], n_rows*sizeof(int), cudaMemcpyHostToDevice));
    AssertCuda(cudaMemcpy(&y_device[0], &y[0], y.size()*sizeof(Number), cudaMemcpyHostToDevice));
    AssertCuda(cudaMemcpy(&x_device[0], &x[0], x.size()*sizeof(Number), cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();
    //Perform Ax = y on GPU
    crsgemvKernel<Number><<<n_blocks, block_size>>>(&A_device[0], &x_device[0], &y_device[0], &row_ptr_device[0], &colIdx_device[0],n_rows);
    cudaDeviceSynchronize();
    // Copy result back to host
    AssertCuda(cudaMemcpy(&y[0], &y_device[0], y.size() * sizeof(Number), cudaMemcpyDeviceToHost));
    //Free on device
    cudaFree(A_device);
    cudaFree(row_ptr_device);
    cudaFree(colIdx_device);
    cudaFree(y_device);
    cudaFree(x_device);
}

template<typename Number>
typename crsMatrix<Number>::sell_c_sigma_data_t crsMatrix<Number>::get_sell_c_sigma(const int C, const int sigma){   
    //C is number of rows in block
    //Sigma is sorting: Assumes sigma=1 - no sorting

    //Number of blocks 
    int n_blocks = (int)ceil( (double)n_rows / (double)C);
    
    //Store largest nnz here
    std::vector<int> largest_nnzs(n_blocks, 0);
    //Now we want to find maximum non-zeros in a row in a block
    for(int i = 0; i < n_rows; i++){
        int start_index = row_ptr[i];
        int end_index = row_ptr[i+1];
        
        
        //nnz is the difference of end index and start index
        int nnz = end_index - start_index;
        
        
        //Each row finds its corresponding block by block = floor(i / C)
        int block_num = i / C;
        largest_nnzs[block_num] = std::max(largest_nnzs[block_num], nnz);

    }
    //Now we know how many elements we need to store in total in SELL-C-sigma
    //and we can compute the offsets for each block
    std::vector<int> sell_c_sigma_offsets(n_blocks+1, 0);

    int total_elems = 0;
    for(int i = 0; i < n_blocks;i++){
        sell_c_sigma_offsets[i] = total_elems;
        total_elems += C * largest_nnzs[i];
    }
    sell_c_sigma_offsets[n_blocks] = total_elems;
        
    //Allocate memory for Sell-C-sigma values and cols
    std::vector<Number> sell_c_sigma_values(total_elems, 0.0); //0 is for padded elements
    std::vector<int> sell_c_sigma_cols(total_elems, -1); //Use -1  as column for padded elements
    
    
    for(int i = 0; i < n_rows; i++){
        int start_index = row_ptr[i];
        int end_index = row_ptr[i+1];
        //Elements in row
        int n_elems = end_index - start_index;
        
        //Each row finds its corresponding block by: block = floor(i / C)
        int block_num = i / C;
        int block_n_cols = largest_nnzs[block_num];
        int block_offset = sell_c_sigma_offsets[block_num];

        // Compute row offset within the block
        int row_offset = i % C;
        
        // Each block has dimensions C x block_n_cols
        for (int j = 0; j < n_elems; j++) {
            // Offset for value and column
            int index = block_offset + row_offset + j * C;

            // Populate SELL-C-sigma value and column
            sell_c_sigma_values[index] = values[start_index + j];
            sell_c_sigma_cols[index] = col_idx[start_index + j];
        }
    
    }
    sell_c_sigma_data_t sell_c_sigma_data;
    sell_c_sigma_data.C = C;
    sell_c_sigma_data.sigma = sigma;
    sell_c_sigma_data.nnz = nnz;
    sell_c_sigma_data.n_rows = n_rows;
    sell_c_sigma_data.n_cols = n_cols;
    sell_c_sigma_data.total_elems = total_elems;
    sell_c_sigma_data.largestnnz = std::move(largest_nnzs);
    sell_c_sigma_data.offset_ptr = std::move(sell_c_sigma_offsets);
    sell_c_sigma_data.col_idx = std::move(sell_c_sigma_cols);
    sell_c_sigma_data.values = std::move(sell_c_sigma_values);
    return sell_c_sigma_data;
    
}
#endif
