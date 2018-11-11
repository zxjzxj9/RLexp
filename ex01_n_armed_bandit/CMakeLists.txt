cmake_minimum_required(VERSION 2.6)

project(ex01_n_armed_bandit)
add_compile_options(-std=c++11)
add_executable(nbandit.x main.cpp bandit.cpp)