#include<vector>
#include<stdexcept>
#include<cassert>
#include<iostream>
#include<cmath>
#include"matrix.h"
#include"crsMatrix.h"
#include"sellCSigmaMatrix.h"

int main(){
    const int N = 16;
    typedef float floatType;

    Matrix<floatType> Delta3d(N*N*N,N*N*N);
    Matrix<floatType>::generate_3d_laplacian(N, Delta3d);

    Matrix<floatType>::csr_data_t csr_data = Delta3d.get_csr_data();   
    
    crsMatrix<floatType> csrDelta(csr_data);

    std::vector<floatType> x(N*N*N, 0.0);
    std::vector<floatType> y_sp(N*N*N, 1.0);
    std::vector<floatType> y_dense(N*N*N, -1.0);
     //Generate random vector
    for(int i = 0;i < N*N*N;i++){
        x[i] = (floatType) rand() / (floatType)RAND_MAX + 10.0; 
    }
    
    //CPU mult
    crsMatrix<floatType>::gemv(csrDelta, x, y_sp);
    Matrix<floatType>::gemv(Delta3d, x, y_dense);
    floatType l2error = 0.0;
    for(int i = 0;i < N*N*N;i++){
        l2error += (y_sp[i] - y_dense[i])*(y_sp[i] - y_dense[i]);
    }
    printf("CPU Error: %.17lf\n", l2error);
    
    //GPU mult
    crsMatrix<floatType>::cuSparsegemv(csrDelta, x, y_sp);
    l2error = 0.0;
    for(int i = 0;i < N*N*N;i++){
        l2error += (y_sp[i] - y_dense[i])*(y_sp[i] - y_dense[i]);
    }
    printf("GPU Error: %.17lf\n", l2error);
 
    return 0;
    
}
