#ifndef __NEURALNETWORK__HPP__
#define __NEURALNETWORK__HPP__

#include <Eigen/Core>
#include <Eigen/StdVector>
using namespace Eigen;

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <bitset>
#include <math.h>
#include <complex>
#include <cmath>
#include <vector>
using namespace std;

class Network{
public:
    Network(int N, int M);
    VectorXd v;
    MatrixXd W;
    VectorXd h;
    VectorXd a;
    VectorXd b;
    Network(const Network& network, VectorXd data);
};

class Gradient{
public:
    Gradient(int N, int M);
    MatrixXd gW;
    VectorXd ga;
    VectorXd gb;
    void calculate(const Network& network);
    Gradient operator+(Gradient& temp);
    Gradient operator/(int N);
    Gradient& operator=(Gradient temp);
};

#endif
