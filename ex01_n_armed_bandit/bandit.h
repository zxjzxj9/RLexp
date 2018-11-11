//
// Created by Victor Zhang on 2018/11/11.
//

#ifndef EX01_N_ARMED_BANDIT_BANDIT_H
#define EX01_N_ARMED_BANDIT_BANDIT_H

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

class Bandit {

public:
    Bandit(int n_arms, float mu = 0.0, float sigma = 1.0):
            n_arms(n_arms), mu(mu), sigma(sigma) {
        std::normal_distribution<float> dist(mu, sigma);
        for(int i=0; i<n_arms; i++) {
            bandit.push_back(dist(rg));
        }
    }

    // Sample from the bandits
    float sample(size_t bandit_idx, float sigma = 1.0) {
        float mu = bandit[bandit_idx];
        std::normal_distribution<float> dist(mu, sigma);
        return dist(rg);
    }

    // Get the argmax of bandits
    size_t get_max() {
        auto max_elem = std::max_element(bandit.begin(), bandit.end());
        return std::distance(std::begin(bandit), max_elem);
    }

    // printout all the element
    void print() {
        std::ostream_iterator<float> out(std::cout, ", ");
        std::copy(std::begin(bandit), std::end(bandit), out);
        std::cout<<std::endl;
    }

private:
    int n_arms;
    float mu;
    float sigma;
    // std::random_device rd;
    //std::mt19937 rg; //{std::random_device{}()};
    std::mt19937 rg{std::random_device{}()};
    // std::normal_distribution<float> dist;
    std::vector<float> bandit;
};

class Policy {

};
#endif //EX01_N_ARMED_BANDIT_BANDIT_H
