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
    Gradient operator-(Gradient& temp);
    Gradient operator/(int N);
    Gradient operator*(double c);
    Gradient& operator=(Gradient temp);
};

Gradient Gradient::operator+(Gradient& temp){
    Gradient result(ga.size(), gb.size());
    result.gW = gW + temp.gW;
    result.ga = ga + temp.ga;
    result.gb = gb + temp.gb;
    return result;
}

Gradient Gradient::operator-(Gradient& temp){
    Gradient result(ga.size(), gb.size());
    result.gW = gW - temp.gW;
    result.ga = ga - temp.ga;
    result.gb = gb - temp.gb;
    return result;
}

Gradient Gradient::operator/(int N){
    Gradient result(ga.size(), gb.size());
    result.gW = gW/N;
    result.ga = ga/N;
    result.gb = gb/N;
    return result;
}

Gradient Gradient::operator*(double c){
    Gradient result(ga.size(), gb.size());
    result.gW = gW*c;
    result.ga = ga*c;
    result.gb = gb*c;
    return result;
}

Gradient& Gradient::operator=(Gradient temp){
    gW = temp.gW;
    ga = temp.ga;
    gb = temp.gb;
    return *this;
}


#endif
