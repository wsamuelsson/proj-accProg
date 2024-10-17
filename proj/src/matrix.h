#ifndef __matrix__
#define __matrix__

#include<vector>
#include<stdexcept>
#include<cassert>
#include<iostream>
#include<cmath>
#include"cblas.h"



template<class Number>
class Matrix{

    
    public:
        //Constructors
        Matrix(int i, int j);
        Matrix(int i, int j, std::vector<Number> entries); 


        //Selectors
        int get_n_rows() const {return n_rows;};
        int get_n_cols() const {return n_rows;};
        const Number * get_static_data_ptr() const {return &data[0];}
        //Methods
        void set_tridiag(std::vector<Number> const &lower, std::vector<Number> const &main, std::vector<Number> const &upper);
        void print_matrix();
        void identity_matrix(int N);
        static void kronecker_product(const Matrix<Number> &A, const Matrix<Number> &B, Matrix<Number> &C);
        static void add(const Matrix<Number> &A, const Matrix<Number> &B, Matrix<Number> &C);
        static void generate_3d_laplacian(const int N, Matrix<Number> &Delta3d);
        static void gemv( const Matrix<Number> &A, const std::vector<Number> &x, std::vector<Number> &y);


        //Structure for csr data 
        typedef struct{
            int nnz; //Non zero elements
            int n_rows; 
            int n_cols;
            std::vector<int> row_ptr; //Row pointers
            std::vector<int> col_idx; //Column index
            std::vector<Number> values; //Non-zero values

        } csr_data_t;

        csr_data_t get_csr_data();
        

    private:
        //Data in row major storage
        std::vector<Number> data;
        int n_rows;
        int n_cols;



};

template<class Number>
void Matrix<Number>::set_tridiag(std::vector<Number> const &lower, std::vector<Number> const &main, std::vector<Number> const &upper) {
    
    // Clear the data vector and resize it to fit the matrix
    data.assign(n_rows * n_cols, 0);

    // Set the main diagonal
    for (int i = 0; i < n_rows; ++i) {
        data[i * n_cols + i] = main[i];
    }

    // Set the lower diagonal
    for (int i = 1; i < n_rows; ++i) {
        data[i * n_cols + (i - 1)] = lower[i - 1];
    }

    // Set the upper diagonal
    for (int i = 0; i < n_rows - 1; ++i) {
        data[i * n_cols + (i + 1)] = upper[i];
    }
}

template<class Number>
Matrix<Number>::Matrix(int i, int j, std::vector<Number> entries){
    data = entries;
    if(entries.size() == i*j){
        n_rows = i;
        n_cols = j;
    }
    else{
        throw std::runtime_error("Error in constructor: Number of matrix entries must be the same as rows*columns");
    }
    
}

template<class Number> 
Matrix<Number>::Matrix(int i, int j){
    assert(i>0 && j>0);
    n_rows = i;
    n_cols = j;
    data.reserve(n_cols * n_rows);
}

template<class Number>
void Matrix<Number>::print_matrix() {
    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            // Accessing element (i, j) in row-major order
            std::cout << data[i * n_cols + j] << "  ";
        }
        std::cout << std::endl;
    }
}

template<class Number>
void Matrix<Number>::kronecker_product(const Matrix<Number> &A, const Matrix<Number> &B, Matrix<Number> &C) {
    int A_rows = A.n_rows, A_cols = A.n_cols;
    int B_rows = B.n_rows, B_cols = B.n_cols;

    // Resize C to the correct size
    C.n_rows = A_rows * B_rows;
    C.n_cols = A_cols * B_cols;
    C.data.resize(C.n_rows * C.n_cols);
    // Compute Kronecker product
    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < A_cols; ++j) {
            Number A_ij = A.data[i * A.n_cols + j]; // Element A(i,j)

            for (int k = 0; k < B_rows; ++k) {
                for (int l = 0; l < B_cols; ++l) {
                    // Element B(k, l)
                    Number B_kl = B.data[k * B.n_cols + l];
                    // Set element C(i * B_rows + k, j * B_cols + l)
                    C.data[(i * B_rows + k) * C.n_cols + (j * B_cols + l)] = A_ij * B_kl;
                }
            }
        }
    }
}

template<class Number>
void Matrix<Number>::identity_matrix(int N){
    // Resize the matrix to N x N
    n_rows = N;
    n_cols = N;
    data.resize(N * N, 0); // Initialize all elements to 0

    // Set the diagonal elements to 1
    for (int i = 0; i < N; ++i) {
        data[i * N + i] = 1.0;
    }
}

template<class Number>
void Matrix<Number>::add(const Matrix<Number> &A, const Matrix<Number> &B, Matrix<Number> &C){
    int rowsInA = A.get_n_rows();
    int colsInA = A.get_n_cols();
    int rowsInB = B.get_n_rows();
    int colsInB = B.get_n_cols();

    if(rowsInA == rowsInB && colsInA == colsInB){
        C.data.resize(colsInA*rowsInA);
        for (int i=0; i < colsInA*rowsInA; i++){
            C.data[i] = A.data[i] + B.data[i];
        }
    }
    else{
        throw std::runtime_error("Error in add: Dimensions must agree.");
    }

}

template<class Number>
void Matrix<Number>::generate_3d_laplacian(const int N, Matrix<Number> &Delta3d){
    if (N > 0){

        Matrix<Number> delta(N,N);
        Matrix<Number> I(N, N);

        //Set up the 1D stencil
        std::vector<Number> main(N, -2.0 / ((Number)(N+1) * (Number)(N+1)));
        std::vector<Number> lower(N-1, 1.0 / ((Number)(N+1) * (Number)(N+1)));
        std::vector<Number> upper(N-1, 1.0 / ((Number)(N+1) * (Number)(N+1)));

        delta.set_tridiag(lower, main, upper);
        I.identity_matrix(N);
        
        //Partial sums and temp variables for matrices
        Matrix<Number> delta3_1(N*N*N, N*N*N);
        Matrix<Number> delta3_2(N*N*N, N*N*N);
        Matrix<Number> delta3_3(N*N*N, N*N*N);
        Matrix<Number> temp(N*N*N, N*N*N);

        //Place holders for kronecker products
        Matrix<Number> delta_kron_I(N*N, N*N);
        Matrix<Number> I_kron_delta(N*N, N*N);
        Matrix<Number> I_kron_I(N*N, N*N);
        

        //Compute first batch of kronecker products
        Matrix<Number>::kronecker_product(delta, I, delta_kron_I);
        Matrix<Number>::kronecker_product(I, delta, I_kron_delta);
        Matrix<Number>::kronecker_product(I, I, I_kron_I);

        //Compute second batch of kronecker products
        Matrix<Number>::kronecker_product(delta_kron_I, I, delta3_1);
        Matrix<Number>::kronecker_product(I_kron_delta, I, delta3_2);
        Matrix<Number>::kronecker_product(I_kron_I, delta, delta3_3);
        
        //Add up
        Matrix<Number>::add(delta3_1, delta3_2, temp);
        Matrix<Number>::add(temp, delta3_3, Delta3d);


    }else{
        throw std::runtime_error("Error in 'generate_3d_laplcian': Laplacian dims must be positive");
    }
    
}

//single precision template specialization
template<>
void Matrix<float>::gemv( const Matrix<float> &A, const std::vector<float> &x, std::vector<float> &y){
    // Ensure the vector dimensions match
    int n_rows = A.get_n_rows();
    int n_cols = A.get_n_cols();

    float alpha = 1.0;
    float beta = 0.0;

    const float *dataptr = A.get_static_data_ptr();
    if (x.size() != n_cols || y.size() != n_rows) {
        throw std::invalid_argument("Vector dimensions do not match matrix dimensions");
    }

    // Use cblas_sgemv for single precision
    cblas_sgemv(CblasRowMajor, CblasNoTrans, n_rows, n_cols, alpha, dataptr, n_cols, x.data(), 1, beta, y.data(), 1);

}

//double precision template specialization
template<>
void Matrix<double>::gemv( const Matrix<double> &A, const std::vector<double> &x, std::vector<double> &y){

    // Ensure the vector dimensions match
    int n_rows = A.get_n_rows();
    int n_cols = A.get_n_cols();

    double alpha = 1.0;
    double beta = 0.0;

    const double *dataptr = A.get_static_data_ptr();
    if (x.size() != n_cols || y.size() != n_rows) {
        throw std::invalid_argument("Vector dimensions do not match matrix dimensions");
    }

    // Use cblas_dgemv for double precision
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n_rows, n_cols, alpha, dataptr, n_cols, x.data(), 1, beta, y.data(), 1);
}


template<class Number>
typename Matrix<Number>::csr_data_t Matrix<Number>::get_csr_data(){
    int nnz = 0;
    for (int i = 0; i < n_cols * n_rows; i++) {
        if (fabs(data[i] - 0.0) >= 1.0e-6) nnz++;
    }
    assert(nnz > 0);
     // Create vectors to store CSR data
    std::vector<Number> values;
    std::vector<int> col_idx;
    std::vector<int> row_ptr(n_rows + 1, 0);  // row_ptr has n_rows+1 entries

    values.reserve(nnz);  
    col_idx.reserve(nnz); 
    int nnz_counter = 0;  // Counter for non-zero values
    
    // Now fill in the CSR data
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            Number val = data[i * n_cols + j];  // Access the matrix in row-major order
            if (fabs(val - 0.0) >= 1.0e-6) {
                values.push_back(val);          //Save non zero value
                col_idx.push_back(j); 
                                                //Keep track of column
                nnz_counter++;
            }
        }
        // After finishing the current row, point to where the next row starts
        row_ptr[i + 1] = nnz_counter;
    }
    
    
    // Construct the csr_data_t struct 

    csr_data_t csr_data;
    csr_data.nnz = std::move(nnz);
    csr_data.n_cols = std::move(n_cols);
    csr_data.n_rows = std::move(n_rows);
    csr_data.values = std::move(values);
    csr_data.col_idx = std::move(col_idx);
    csr_data.row_ptr = std::move(row_ptr);

    return csr_data;
}



#endif