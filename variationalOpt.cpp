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

vector<VectorXd> Metropolis_sampling(const Network& network, int M, int steps){
    vector<VectorXd> samples;
    Network next(network, VectorXd::Ones(network.v.size()));
    double prev_prob = Boltzmann_probability(next, 1);
    int flip;
    double r;
    double Acceptance;
    for (int i=0; i<M; i++){
        for (int j=0; j<steps; j++){
            flip = rand()%network.v.size();
            next.v(flip) = -next.v(flip) + 1;
            r = (double)rand()/RAND_MAX;
            Acceptance = Boltzmann_probability(next, 1)/prev_prob;
            next.v(flip) = (Acceptance > r)? next.v(flip) : (-next.v(flip)+1);
            prev_prob = Boltzmann_probability(next, 1);
        }
        samples.push_back(next.v);
        next.v = VectorXd::Ones(next.v.size());
        prev_prob = Boltzmann_probability(next, 1);
    }
    return samples;
}
    

//1d Transverse Field Ising Model
double local_energy(double Hamiltonian[2], const Network& network){
    double energy = 0;
    double J = Hamiltonian[0];
    double B = Hamiltonian[1];
    for (int i=0; i<network.v.size(); i++){
        energy += (-J)*4*(network.v(i)-0.5)*(network.v((i+1)%network.v.size())-0.5);
    }
    energy *= sqrt(Boltzmann_probability(network, 1));
    Network temp_net(network, network.v);
    for (int k=0; k<network.v.size(); k++){
        temp_net.v(k) = -network.v(k) + 1;
        energy += (-B)*sqrt(Boltzmann_probability(temp_net, 1));
        temp_net.v(k) = network.v(k);
    }
    energy /= sqrt(Boltzmann_probability(network, 1));
    return energy;
}

Gradient grad_energy(Network network, double Hamiltonian[2], const vector<VectorXd>& samples){
    Gradient avg2(network.v.size(), network.h.size());
    Gradient avg1(network.v.size(), network.h.size());
    Gradient temp(network.v.size(), network.h.size());
    double avg_local_energy = 0;
    for (int i=0; i<samples.size(); i++){
        network.v = samples[i];
        temp.calculate(network);
        avg1 = avg1 - temp;
        avg_local_energy += local_energy(Hamiltonian, network);
        temp = temp*local_energy(Hamiltonian, network);
        avg2 = avg2 - temp;
    }
    avg1 = avg1/samples.size();
    avg_local_energy /= samples.size();
    avg2 = avg2/samples.size();
    avg1 = avg1*avg_local_energy;
    return avg2 - avg1;
}

void descent(Network& network, double rate, Gradient& grad_energy){
    network.W -= rate * grad_energy.gW;
    network.b -= rate * grad_energy.gb;
    network.a -= rate * grad_energy.ga;
}

void storeResult(VectorXd& result, string filename){
    ofstream outFile;
    outFile.open(filename);
    outFile<<result;
    outFile.close();
}

int main(){
    string parameter;  
    int spins;
    cin>>parameter>>spins;
    int hidden;
    cin>>parameter>>hidden;
    Network network(spins, hidden);
    network.v(0) = 1;
    
    int M; //mini batch
    cin>>parameter>>M;
    int steps;
    cin>>parameter>>steps;
    vector<VectorXd> samples = Metropolis_sampling(network, M, steps);
    double learning_rate;
    cin>>parameter>>learning_rate;
    double Hamiltonian[2];
    cin>>parameter>>Hamiltonian[0]>>Hamiltonian[1]; //J, B
    Gradient force = grad_energy(network, Hamiltonian, samples);
    double error = (force.gW).maxCoeff();
    double energy = local_energy(Hamiltonian, network);
    cout<<energy<<"\t"<<error<<endl;
    double threshold;
    cin>>parameter>>threshold;
    double KL_div = 0;
    while (error > threshold){ //update parameters
        descent(network, learning_rate, force);
        samples = Metropolis_sampling(network, M, steps);
        force = grad_energy(network, Hamiltonian, samples);
        energy = 0;
        for (int i=0; i<samples.size(); i++){
            network.v = samples[i];
            energy += local_energy(Hamiltonian, network);
        }
        energy /= samples.size();
        error = force.gW.maxCoeff();
        error = max(error, force.ga.maxCoeff());
        error = max(error, force.gb.maxCoeff());
        cout<<energy<<"\t"<<error<<endl; 
    }

    int dim = 1<<spins;
    VectorXd prob(dim);
    cout<<"storing results..."<<endl;
    for (int i=0; i<dim; i++){
        for (int j=0; j<spins; j++){
            network.v(j) = (int)(i % (1<<(spins-j))) / (1<<(spins-1-j));
        }
        prob(i) = Boltzmann_probability(network, 1);
    }
    double Z = prob.sum();
    prob = prob/Z;
    storeResult(prob, "Probability");
    
    return 0;
}
    
