//
// Created by Victor Zhang on 2018/11/21.
//

#ifndef EX02_GRIDWORLD_GRIDWORLD_H
#define EX02_GRIDWORLD_GRIDWORLD_H

#include <iostream>
#include <memory>
#include <map>

/* Suppose the gridworld has a size of nxn
 * the reward of going to each grid is given in a map <pair<x, y>, val>
 * where x is the row, y is the column , val is the reward value
 * special teleportation rules are also encoded in a map <pair<x1, y1>, pair<x2, y2>>
 * where a bot went ot x1, y1 will be transported to x2, y2
 */

class GridWorld {

public:
    GridWorld(int n, std::map<std::pair<int, int>, float> reward,
              std::map<std::pair<int, int>, std::pair<int, int> > tele
        ): n(n) {

    }

private:
    std::unique_ptr<float []> reward;
    std::unique_ptr<int []> tele;
    int n;
};


#endif //EX02_GRIDWORLD_GRIDWORLD_H
