//
// Created by Victor Zhang on 2018/11/11.
//
/**
 * This code is aimed at simulating a n-armed bandit
 */

#include <iostream>
#include <memory>
#include "bandit.h"

int main(int argc, char** argv) {
    // Bandit bd(10);
    // bd.print();
    // std::cout<<bd.get_max()<<std::endl;
    std::unique_ptr<Policy> p(new GreedyPolicy(2000, 10));
    p->simulate(100);
    return 0;
}
