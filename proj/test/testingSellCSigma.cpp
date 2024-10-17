#include<vector>
#include<stdexcept>
#include<cassert>
#include<iostream>
#include<cmath>
#include"matrix.h"
#include"crsMatrix.h"
#include"sellCSigmaMatrix.h"

int main(){

    const int N = 4;
    const int M = 5;
    std::vector<float> matrixData = {1.0, 0.0, 2.0, 0.0, 
                                    0.0, 3.0, 0.0, 0.0, 
                                    0.0, 0.0, 0.0, 0.0, 
                                    4.0, 5.0, 0.0, 0.0, 
                                    0.0, 6.0, 7.0, 8.0};
    Matrix<float> A(M, N, matrixData);

    Matrix<float>::csr_data_t csr_data = A.get_csr_data();
    int nnz = csr_data.nnz;

    crsMatrix<float> A_sp(csr_data);

    A_sp.print_csr();
    crsMatrix<float>::sell_c_sigma_data_t sell_c_sigma_data;
    sell_c_sigma_data = A_sp.get_sell_c_sigma(2, 1);
    
    sellCSigmaMatrix<float>  A_sellCSigma(sell_c_sigma_data);
    printf("%d\n", A_sellCSigma.get_total_elems());
    return 0;
}