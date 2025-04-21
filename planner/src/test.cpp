#include "grid_map.h"
#include "penal_traj_opt.h"

#include <iostream>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int main(){
    std::unordered_map<std::string, double> grid_map_parms = { {"map_size_x", 12.0}, {"map_size_y", 10.0}, {"resolution", 0.1 }};

    GridMap map;
    map.init(grid_map_parms);

    RowMatrixXd sdf_map = map.getMap();

    int r = sdf_map.rows();
    int c = sdf_map.cols();
    std::cout << "sdf map size" << r << " " << c << std::endl;
    RowMatrixXi b;

    return 0;
}