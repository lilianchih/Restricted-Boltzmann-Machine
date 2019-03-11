#include "neuralNetwork.hpp"

Network::Network(int N, int M){
    v = VectorXd::Zero(N);
    W = MatrixXd::Random(N, M);
    a = VectorXd::Random(N);
    b = VectorXd::Random(M);
    h = VectorXd::Zero(M);
}

Network::Network(const Network& network, VectorXd data){
    v = data;
    W = network.W;
    a = network.a;
    b = network.b;
    h = network.h;
}

Gradient::Gradient(int N, int M){
    gW = MatrixXd::Zero(N, M);
    ga = VectorXd::Zero(N);
    gb = VectorXd::Zero(M);
}

VectorXd Sigmoid(VectorXd x){
    VectorXd exponential = (-x.array()).exp();
    return (exponential + VectorXd::Ones(x.size())).cwiseInverse();
}

void Gradient::calculate(const Network& network){
    VectorXd theta = network.W.transpose()*network.v + network.b;
    gW = -network.v*Sigmoid(theta).transpose();
    gb = -Sigmoid(theta);
    ga = -network.v;
}

double Boltzmann_probability(const Network& network, double partition){
    VectorXd theta = network.W.transpose()*network.v + network.b;
    VectorXd exponential = theta.array().exp();
    return exp(network.v.dot(network.a))*(VectorXd::Ones(theta.size())+exponential).prod()/partition;
}

double Partition_function(Network network, int N){
    double output = 0;
    int dim = 1 << N;
    for (int i=0; i<dim; i++){
        for (int j=0; j<N; j++){
            network.v(j) = (int)(i % (1<<(N-j))) / (1<<(N-1-j));
        }
        output += Boltzmann_probability(network, 1.0);
    }
    return output;
}

void update_h(Network& network){
    VectorXd cond_prob = Sigmoid(network.W.transpose()*network.v + network.b);
    VectorXd r = 0.5*(VectorXd::Random(network.h.size()) + VectorXd::Ones(network.h.size()));
    for (int i=0; i<network.h.size(); i++){
        network.h(i) = (r(i) > cond_prob(i))? 0:1;
    }
}

void update_v(Network& network){
    VectorXd cond_prob = Sigmoid(network.W*network.h + network.a);
    VectorXd r = 0.5*(VectorXd::Random(network.v.size()) + VectorXd::Ones(network.v.size()));
    for (int i=0; i<network.v.size(); i++){
        network.v(i) = (r(i) > cond_prob(i))? 0:1;
    }
}

Gradient grad_data(Network network, vector<VectorXd>& data){
    Gradient avg(network.v.size(), network.h.size());
    Gradient temp(network.v.size(), network.h.size());
    for (int i=0; i<data.size(); i++){
        network.v = data[i];
        temp.calculate(network);
        avg = avg + temp;
    }
    avg = avg/data.size();
    return avg;
}

Gradient grad_sim(Network& network, vector<VectorXd>& data, int M, int K){
    Gradient avg(network.v.size(), network.h.size());
    Gradient temp_grad(network.v.size(), network.h.size());
    for (int m=0; m<M; m++){
        //int r = rand() % data.size();
        int r = m;
        Network temp_net(network, data[r]);
        for (int k=0; k<K; k++){
            update_h(temp_net);
            update_v(temp_net);
        }
        temp_grad.calculate(temp_net);
        avg = avg + temp_grad;
    }
    avg = avg/M;
    return avg;
}

void descent(Network& network, double rate, Gradient& grad_data, Gradient& grad_sim){
    network.W -= rate * (grad_data.gW - grad_sim.gW);
    network.b -= rate * (grad_data.gb - grad_sim.gb);
    network.a -= rate * (grad_data.ga - grad_sim.ga);
}

vector<VectorXd> readData(string filename, int N){
    vector<VectorXd> data;
    ifstream indata;
    indata.open(filename);
    
    VectorXd temp(N);
    while(!indata.eof()){
        for (int i=0; i<N; i++){
            indata >> temp(i);
        }
        data.push_back(temp);
    }
    return data;
}

int main(){  
    int spins = 10;
    int hidden = 40;
    Network network(spins, hidden);
    cout<<"network spins=10 hidden=40"<<endl;
    vector<VectorXd> data = readData("data.txt", spins);
    cout<<"\"data.txt\" size: "<<data.size()<<endl;    

    int K = 5; //CD-k
    int M = (data.size() > 10000)? 10000:data.size(); //mini batch
    double learning_rate = 0.01;
    Gradient data_avg = grad_data(network, data);
    Gradient sim_avg = grad_sim(network, data, M, K);
    double error = (data_avg.gW - sim_avg.gW).maxCoeff(); 
    double KL_div = 0;
    while (error > 0.01){ //update parameters
        descent(network, learning_rate, data_avg, sim_avg);
        data_avg = grad_data(network, data);
        sim_avg = grad_sim(network, data, M, K);
        error = (data_avg.gW - sim_avg.gW).maxCoeff();
        error = max(error, (data_avg.ga - sim_avg.ga).maxCoeff());
        error = max(error, (data_avg.gb - sim_avg.gb).maxCoeff());
        cout<<error<<endl; 
    }

    double Z = Partition_function(network, spins);
    network.v = VectorXd::Zero(spins);
    double prob = Boltzmann_probability(network, Z);
    cout<<"Probability in all |0> "<<prob<<endl;
    network.v = VectorXd::Ones(spins);
    prob = Boltzmann_probability(network, Z);
    cout<<"Probability in all |1> "<<prob<<endl;
     
        
    return 0;
}
    
