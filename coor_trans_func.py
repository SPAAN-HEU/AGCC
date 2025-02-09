from numpy import *


def coor_tans(coor_lidar:ndarray,trans_imu2world:ndarray,trans_velo2imu:ndarray)->ndarray:

    _coor_imu = dot(trans_velo2imu,coor_lidar.T)

    _coor_world = dot(trans_velo2imu, _coor_imu)

    return _coor_world[0:3]

