from numpy import *
basedir = "D:\\Data\\Loop\\dataset"
# basedir = "/media/jmh/DATA/Data/Loop/dataset"
#00
date = '2011_10_03'
drive = '0027'
#05
date5 = '2011_09_302'
drive5 = '0018'
# #02
# date = '2011_10_032'
# drive = '0034'
#08
date8 = '2011_09_303'
drive8 = '0028'
# #07
# date = '2011_09_301'
# drive = '0027'

sequence = ['00','01','02','03','04','05','06','07','08','09','10']
earth_R = 6378137
f = 1/298.257222101
thred = 245 #回环检测最终阈值
rms_thred = 1
frame_thred = 100
distance_thred =4.5
centroid_car = array([]).reshape(4,-1)
size_car = array([])
centroid_bicycle = array([]).reshape(4,-1)
size_bicycle = array([])
centroid_bus = array([]).reshape(4,-1)
size_bus = array([])
centroid_motorcycle = array([]).reshape(4,-1)
size_motorcycle = array([])
centroid_person = array([]).reshape(4,-1)
size_person = array([])
centroid_trunk = array([]).reshape(4,-1)
size_trunk = array([])
centroid_pole = array([]).reshape(4,-1)
size_pole = array([])
centroid_traffic_sign = array([]).reshape(4,-1)
size_traffic_sign = array([])
loop_closure = array([]).reshape(-1, 2)
search_range = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
