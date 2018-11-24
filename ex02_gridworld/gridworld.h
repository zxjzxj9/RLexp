//
// Created by Victor Zhang on 2018/11/21.
//

#ifndef EX02_GRIDWORLD_GRIDWORLD_H
#define EX02_GRIDWORLD_GRIDWORLD_H

#include <iostream>
#include <memory>
#include <map>
#include <exception>
/* Suppose the gridworld has a size of nxm
 * the reward of going to each grid is given in a map <pair<x, y>, val>
 * where x is the row, y is the column , val is the reward value
 * special teleportation rules are also encoded in a map <pair<x1, y1>, pair<x2, y2>>
 * where a bot went ot x1, y1 will be transported to x2, y2
 */
enum Direction {
    LEFT,
    RIGHT,
    UP,
    DOWN,
};

class GridWorld {

public:
    GridWorld(int n, int m, std::map<std::pair<int, int>, float> reward,
              std::map<std::pair<int, int>, std::pair<int, int> > teleport
        ): n(n), m(m) {
        for(int i=0; i<n; i++) {
            for(int j=0; j<m; j++) {
                this->reward[i*m + j] = reward[std::pair<int, int>(i, j)];
                auto target = teleport[std::pair<int, int>(i, j)];
                this->teleport[i*m + j] = target.first*m + target.first;
            }
        }
    }

    // step a move, return the reward, and teleport
    float move(Direction d) {
        switch(d) {
            case LEFT:
                break;
            case RIGHT:
                break;
            case UP:
                break;
            case DOWN:
                break;
            default:
                throw std::runtime_error("No such dircetion!");
                //break;
        }
        return 0;
    }

private:
    std::unique_ptr<float []> reward;
    std::unique_ptr<int []> teleport;
    int n,m;
};


#endif //EX02_GRIDWORLD_GRIDWORLD_H
