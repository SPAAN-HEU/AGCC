from numpy import *
import pykitti
import os

class KittiData(object):
    def __init__(self, basedir:str, date:int, drive:int,sequence:int):
        self.basedir = basedir
        self.date = date
        self.drive = drive
        self.raw_data = pykitti.raw(basedir, date, drive)
        self.sequence = sequence
        self.odometry = pykitti.odometry(basedir,sequence)
    def get_file_num(self):
        return len(os.listdir(self.basedir + "/sequences/"+self.sequence+"/velodyne"))
    def get_velo_data(self,velo_index:int):
        return self.odometry.get_velo(velo_index)[:,0:3]
    def get_T_world2imu(self,imu_index:int):
        #由IMU坐标系转换至世界坐标系
        return self.raw_data.oxts[imu_index].T_w_imu
    def get_T_imu2velo(self, velo_index:int):
        #由IMU坐标系转换至雷达坐标系
        return self.raw_data.calib.T_velo_imu
    def get_labels(self,labels_index:int):
        _labels = fromfile(self.basedir + "/sequences/"+self.sequence+"/labels/"+str(labels_index).zfill(6)+".label", dtype=uint32).reshape((-1))
        _semantic_label = _labels & 0xFFFF
        _instance_label = _labels >> 16
        return _semantic_label,_instance_label
    def get_pos(self,pos_index:int)->ndarray:
        return self.raw_data.oxts[pos_index].T_w_imu[0:3,-1]

