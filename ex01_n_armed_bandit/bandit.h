//
// Created by Victor Zhang on 2018/11/11.
//

#ifndef EX01_N_ARMED_BANDIT_BANDIT_H
#define EX01_N_ARMED_BANDIT_BANDIT_H

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>

class Bandit {

public:
    Bandit(int n_arms, float mu = 0.0, float sigma = 1.0):
            n_arms(n_arms), mu(mu), sigma(sigma) {
        std::normal_distribution<float> dist(mu, sigma);
        for(int i=0; i<n_arms; i++) {
            bandit.push_back(dist(rg));
            reward.push_back(0);
            cnt.push_back(0);
        }
        reward_tot = 0;
    }

    // Sample from the bandits, get a reward
    void sample(size_t bandit_idx, float sigma = 1.0) {
        cnt_tot += 1;
        float mu = bandit[bandit_idx];
        std::normal_distribution<float> dist(mu, sigma);
        float ret = dist(rg);
        // Update reward for every bandit number
        // cnt[bandit_idx] += 1;
        update(ret, bandit_idx);
        reward_tot += ret;
    }

    // Update reward for each step
    virtual void update(float val, int bandit_idx) {
        cnt[bandit_idx] += 1;
        reward[bandit_idx] = reward[bandit_idx] +
                             (val - reward[bandit_idx])/static_cast<float>(cnt[bandit_idx]);
    }

    // Get the argmax of bandits
    size_t get_max_bandit_loc() {
        auto max_elem = std::max_element(std::begin(bandit), std::end(bandit));
        return std::distance(std::begin(bandit), max_elem);
    }

    size_t get_max_reward_loc() {
        auto max_elem = std::max_element(std::begin(reward), std::end(reward));
        return std::distance(std::begin(reward), max_elem);
    }

    size_t get_arms_num() {
        return n_arms;
    }

    float get_avg_reward() {
        return reward_tot/static_cast<float>(cnt_tot);
    }

    // printout all the element
    void print() {
        std::ostream_iterator<float> out(std::cout, ", ");
        std::copy(std::begin(bandit), std::end(bandit), out);
        std::cout<<std::endl;
    }

private:
    int n_arms;
    int cnt_tot;
    float mu;
    float sigma;
    float reward_tot;
    std::mt19937 rg{std::random_device{}()};
    std::vector<float> bandit;
    std::vector<float> reward;
    std::vector<int> cnt;
    //std::vector<int> cnt;
};

// Define a basic policy class to simulate the bandit
class Policy {
public:
    //Policy(const Bandit& bandit): bandit(bandit) {}
    virtual void simulate(int n) = 0;
    virtual ~Policy(){}
private:
    // Perform a single run, return #1: average reward, #2: average hit rate
    virtual std::pair<float, float> step() = 0;
};


// Define greedy RL policy
class GreedyPolicy: public Policy {
public:
    GreedyPolicy(int nagents, int narms):
            nagents(nagents), narms(narms) {
        for(int i=0; i<nagents; i++) agent.emplace_back(narms);
    }

    std::pair<float, float> step() {
        float reward_tot = 0;
        int hit_cnt = 0;
        for(int i=0; i<nagents; i++) {
            int max = agent[i].get_max_reward_loc();
            //std::cout<<max<<std::endl;
            agent[i].sample(max);
            reward_tot += agent[i].get_avg_reward();
            if (agent[i].get_max_reward_loc() == agent[i].get_max_bandit_loc()) hit_cnt++;
        }
        return std::make_pair<float, float>(reward_tot/static_cast<float>(nagents),
                                            hit_cnt/ static_cast<float>(nagents));
    }

    void simulate(int max_step) {
        for(int i=0; i< max_step; i++) {
            auto ret = step();
            std::cout<<"Current Step: "<<std::setw(6)<<i+1
                     <<" , Average Reward: "<<std::setw(10)<< ret.first
                     <<" , Average Hit Rate: " << std::setw(10) << ret.second
                     <<std::endl;
        }
    }

private:
    std::vector<Bandit> agent;
    int nagents;
    int narms;
};

class EGreedyPolicy: public Policy {
public:
    EGreedyPolicy(int nagents, int narms, float epsilon=0.1):
            nagents(nagents), narms(narms), epsilon(epsilon) {
        for(int i=0; i<nagents; i++) agent.emplace_back(narms);
    }

    std::pair<float, float>step() {
        float reward_tot = 0;
        int hit_cnt = 0;
        for(int i=0; i<nagents; i++) {
            int max;
            if (dist(rg) > epsilon) {
                max = agent[i].get_max_reward_loc();
            } else {
                // Randomly choice between 0 ~ max
                max = static_cast<int>(dist(rg)*agent[i].get_arms_num());
            }
            //std::cout<<max<<std::endl;
            agent[i].sample(max);
            reward_tot += agent[i].get_avg_reward();
            if (agent[i].get_max_reward_loc() == agent[i].get_max_bandit_loc()) hit_cnt++;
        }
        return std::make_pair<float, float>(reward_tot/static_cast<float>(nagents),
                                            hit_cnt/ static_cast<float>(nagents));
    }


    void simulate(int max_step) {
        for(int i=0; i< max_step; i++) {
            auto ret = step();
            std::cout<<"Current Step: "<<std::setw(6)<<i+1
                     <<" , Average Reward: "<<std::setw(10)<< ret.first
                     <<" , Average Hit Rate: " << std::setw(10) << ret.second
                     <<std::endl;
        }
    }

private:
    std::vector<Bandit> agent;
    int nagents;
    int narms;
    float epsilon;
    std::mt19937 rg{std::random_device{}()};
    std::uniform_real_distribution<> dist{0.0, 1.0};
};

#endif //EX01_N_ARMED_BANDIT_BANDIT_H
