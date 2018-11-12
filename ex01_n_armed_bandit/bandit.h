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
        reward = 0;
        count = 0;
    }

    // Sample from the bandits, get a reward
    float sample(size_t bandit_idx, float sigma = 1.0) {
        float mu = bandit[bandit_idx];
        std::normal_distribution<float> dist(mu, sigma);
        float ret = dist(rg);
        reward += ret;
        count += 1;
        return ret;
    }

    // Get average reward
    float avg_reward() {
        if (count == 0) return 0.0;
        return reward/ static_cast<float>(count);
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
    float reward;
    int count;
    // std::random_device rd;
    //std::mt19937 rg; //{std::random_device{}()};
    std::mt19937 rg{std::random_device{}()};
    // std::normal_distribution<float> dist;
    std::vector<float> bandit;
};

// Define a basic policy class to simulate the bandit
class Policy {

public:
    //Policy(const Bandit& bandit): bandit(bandit) {}
    virtual void simulate(int n) = 0;

private:
    // Perform a single run
    virtual void run(int n) = 0;

};


// Define greedy RL policy
class GreedyPolicy: Policy {

};

#endif //EX01_N_ARMED_BANDIT_BANDIT_H
