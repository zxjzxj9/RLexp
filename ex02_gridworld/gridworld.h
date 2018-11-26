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
        // initialize all the reward to 0, teleport to -1 (means no teleportation)
        std::fill(this->reward.get(), this->reward.get() + n*m - 1, 0);
        std::fill(this->teleport.get(), this->teleport.get() + n*m - 1, -1);
        for(int i=0; i<n; i++) {
            for(int j=0; j<m; j++) {
                this->reward[i*m + j] = reward[std::pair<int, int>(i, j)];
                auto target = teleport[std::pair<int, int>(i, j)];
                this->teleport[i*m + j] = target.first*m + target.first;
            }
        }
    }

    // step a move, return the reward, and teleport
    // always set out-of-boundary reward -1
    float move(Direction d, std::pair<int, int>& coord) {
        int coordt = coord.first*m + coord.second;
        switch(d) {
            case LEFT:
                if(coord.first == 0) return -1;
                coord.first -= 1;
                break;
            case RIGHT:
                if(coord.first == m - 1) return -1;
                break;
            case UP:
                if(coord.second == 0) return -1;
                break;
            case DOWN:
                if(coord.second == n - 1) return -1;
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

    // convert the coordinate x, y to array index
    int coord2idx(std::pair<int, int> coord) {

    }

    // inverse of the above function
    std::pair<int, int> idx2coord(int idx) {

    }

    int n,m;
};


#endif //EX02_GRIDWORLD_GRIDWORLD_H
