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
#include <cmath>

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
    virtual void sample(size_t bandit_idx, float sigma = 1.0) {
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

protected:
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


// Gradient based Bandit
class GBandit: public Bandit {
public:
    GBandit(int n_arms, float mu = 0.0, float sigma = 1.0, float lr=0.1):
            Bandit(n_arms, mu, sigma), lr(lr) {
        for(int i=0; i<n_arms; i++) {
            H.push_back(0);
        }
    }

    // return sampled index according to saved H
    int sample_idx() {
        //std::unique_ptr<float[]> prob{new float[Bandit::n_arms]};
        auto prob = std::unique_ptr<float[]>{new float[Bandit::n_arms]};
        std::copy(std::begin(H), std::end(H), prob.get());
        std::transform(prob.get(), prob.get()+Bandit::n_arms, prob.get(), [=](float x)->float{return exp(x);});
        float sum = std::accumulate(prob.get(), prob.get()+Bandit::n_arms, 0);
        // Normalize
        std::transform(prob.get(), prob.get()+Bandit::n_arms, prob.get(), [=](float x)->float{return x/sum;});
        // https://stackoverflow.com/questions/4437250/choose-list-variable-given-probability-of-each-variable
        // Sample from distribution
        std::uniform_real_distribution<float> dist(0.0, 1.0);
        float p = dist(Bandit::rg);
        for(int i=0; i<Bandit::n_arms; i++) {
            p -= prob[i];
            if(p<=0) {
                probt = prob[i];
                return i;
            }
        }
        probt = prob[0];
        return 0;
    }

    void sample(float sigma=1.0) {
        cnt_tot += 1;
        size_t bandit_idx = sample_idx();
        float mu = bandit[bandit_idx];
        std::normal_distribution<float> dist(mu, sigma);
        float ret = dist(rg);
        update(ret, bandit_idx);
        reward_tot += ret;
    }

    virtual void update(float val, int bandit_idx) {
        float avg_reward = get_avg_reward();
        cnt[bandit_idx] += 1;
        reward[bandit_idx] = reward[bandit_idx] +
                             (val - reward[bandit_idx])/static_cast<float>(cnt[bandit_idx]);
        H[bandit_idx] = H[bandit_idx] + lr*(val - avg_reward)*(1-probt);
        for(int i=0; i<Bandit::n_arms; i++) {
            if(i == bandit_idx) continue;
            H[i] = H[i] - lr*(val - avg_reward)*probt;
        }
    }



private:
    // define learning rate
    float lr;
    float probt;
    std::vector<float> H;
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
