from coor_trans_func import *
from load_data import *
from global_resource import *
from utils import *
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import numpy as np
import time
icp_res = array([])
num_trunk = array([])
num_pole = array([])
res_now = 0
res_last = 0
times = 0
Kitti = KittiData(basedir,date5,drive5,sequence[5])
labels = get_labels_from_yaml("./config/semantic-kitti.yaml")
print("数据加载完成")
start = time.time()
for i in range(Kitti.get_file_num()):
    velo_data = Kitti.get_velo_data(i)
    velo_data = insert(velo_data, 3, 1, axis=1).T
    semantic_labels,instance_labels = Kitti.get_labels(i)
    if i <= 1000:
        centroid_trunk,num_t = static_centroidV2(velo_data,labels.get('trunk'),semantic_labels,size_trunk,centroid_trunk,linalg.inv(Kitti.get_T_imu2velo(i)),Kitti.get_T_world2imu(i+1100),i)
        num_trunk = np.append(num_trunk,num_t)
        centroid_pole,num_p = static_centroidV2(velo_data,labels.get("pole"),semantic_labels,size_pole,centroid_pole,linalg.inv(Kitti.get_T_imu2velo(i)),Kitti.get_T_world2imu(i+1100),i)
        num_pole = np.append(num_pole,num_p)
        # print("第",i,"次完成")
        continue
    tree_pole = KDTree(centroid_pole.T[0:(i-600),0:3])
    tree_trunk = KDTree(centroid_trunk.T[0:(i-600),0:3])
    candidate_pole,num_p = loop_centroidV2(velo_data, semantic_labels, labels.get("pole"), linalg.inv(Kitti.get_T_imu2velo(i)), Kitti.get_T_world2imu(i+1100),i)
    num_pole = np.append(num_pole,num_p)
    candidate_trunk,num_t = loop_centroidV2(velo_data, semantic_labels, labels.get('trunk'), linalg.inv(Kitti.get_T_imu2velo(i)), Kitti.get_T_world2imu(i+1100),i)
    num_trunk = np.append(num_trunk,num_t)
    if len(candidate_pole.shape) != 1 or len(candidate_trunk.shape) != 1:
        print("第", i, "帧未发现回环")
        centroid_pole = append(centroid_pole,zeros((4,1)),axis=1)
        centroid_trunk = append(centroid_trunk,zeros((4,1)),axis=1)
        continue
    distance_pole,index_pole = tree_pole.query(candidate_pole[0:3],k=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])
    distance_trunk,index_trunk = tree_trunk.query(candidate_trunk[0:3],k=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])
    #测试
    # index_pole_test = np.min(index_pole)
    # index_trunk_test = np.min(index_trunk)
    #测试结束

    
    # if abs(index_pole+res_now+1 - i) <= frame_thred and abs(index_trunk+res_now+1 - i) <= frame_thred:
    #     print("第", i, "帧未发现回环")
    #     centroid_trunk = append(centroid_trunk,candidate_trunk.reshape(-1,1),axis=1)
    #     centroid_pole = append(centroid_pole, candidate_pole.reshape(-1,1), axis=1)
    #     continue
    candidate = append(index_pole,index_trunk)
    candidate = unique(candidate)
    if i >= 1560:
        candidate
    if candidate.shape[0] == 1:
        _score_distance = distance_compute(Kitti,i,candidate[0])
        if _score_distance <= distance_thred:
            res_now = candidate[0]
            loop_closure = append(loop_closure,array([candidate[0],i]).reshape(-1,2),axis=0)
            icp_res = append(icp_res,array([candidate[0],i,2]))
            t1 = position_distance_compute1(i,candidate[0],Kitti)
            t1
    else:   
        distance = array([])
        for k in candidate:
            _score_distance = position_distance_compute1(i+1100,k+1100,Kitti)
            distance = append(distance,_score_distance)
        if distance.min() < distance_thred:
            T_w_current = Kitti.get_T_world2imu(i+1100)@np.linalg.inv(Kitti.get_T_imu2velo(i))
            T_w_loop = Kitti.get_T_world2imu(candidate[argmin(distance)]+1100)@np.linalg.inv(Kitti.get_T_imu2velo(candidate[argmin(distance)]))
            T_loop_current = T_w_current@np.linalg.inv(T_w_loop)
            velo_current = velo_data.T 
            res_now = candidate[argmin(distance)]
            velo_loop = Kitti.get_velo_data(res_now)
            velo_loop = insert(velo_loop, 3, 1, axis=1)
            velo_current_loop = (T_loop_current@velo_current.T).T
            # if velo_current_loop .shape[0] <= velo_loop.shape[0]:
                #这里是就方法，如果不行，只保留这一部分，剩下的全部删掉
            velo_loop_tree = KDTree(velo_loop)
            d_c,i_c = velo_loop_tree.query(velo_current_loop)
            #velo_cal中的所有点与velo_loop中的点一一对应
            velo_cal = velo_loop[i_c]
            sigma_x,sigma_y,sigma_z = sigma_estimate2(velo_current_loop,velo_cal,T_loop_current[:3,:3])
            weight = np.ones((3,velo_cal.shape[0]))
            weight[0,1:] = sigma_x[1:]/sigma_x[0]
            weight[1,1:] = sigma_y[1:]/sigma_y[0]
            weight[2,1:] = sigma_z[1:]/sigma_z[0]
            error = weight.T*(velo_current_loop-velo_cal)[:,0:3]
            d = error**2
            d = np.sqrt(d[:,0]+d[:,1]+d[:,2])
            loss = np.sum(d)/d_c.shape[0]
            loss2 = np.sum(d)/np.sum(np.sqrt(weight[0]**2+weight[1]**2+weight[2]**2))
            # else:
            #     velo_current_loop_tree = KDTree(velo_current_loop)
            #     d_c,i_c = velo_current_loop_tree.query(velo_loop)
            #     velo_cal = velo_current_loop[i_c]
            #     d = (velo_cal-velo_loop)**2
            loop_closure = append(loop_closure,array([candidate[argmin(distance)],i]).reshape(-1,2),axis=0)
            icp_res = append(icp_res,array([candidate[argmin(distance)],i,min(distance),loss]))
            t1 = position_distance_compute1(i,candidate[argmin(distance)],Kitti)
            if i > 1620:
                t1
    centroid_trunk = append(centroid_trunk, candidate_trunk.reshape(-1,1), axis=1)
    centroid_pole = append(centroid_pole, candidate_pole.reshape(-1,1), axis=1)
    print(i)
    # if res_last != res_now:
    #     res_last = res_now
    # else:
    #     times += 1
    # if i >= 100 and times >= 10:
    #     times = 0
    #     res_last = res_now = 0

    # if 4451<=i<=4528 or 1570 <=i<=1635 or 2446<=i<=2460 or 3295<=i<=3844:
    #     with open("./Test_data/score.txt","a") as f:
    #         savetxt(f,candidate.reshape(1,-1),fmt="%d")
    #         savetxt(f,append(score,i).reshape(1,-1),fmt="%.2f")
#     if score.size == 0 or score.min() > thred:
#         print("第", i, "帧未发现回环")
#         centroid_trunk = append(centroid_trunk, candidate_trunk, axis=1)
#         centroid_pole = append(centroid_pole, candidate_pole, axis=1)
#         size_trunk = append(size_trunk, size_trunk[-1]+candidate_trunk.shape[1]).astype(int)
#         size_pole = append(size_pole, size_pole[-1]+candidate_pole.shape[1]).astype(int)
#         continue
#     index = candidate[argmin(score)]
#     loop_closure = append(loop_closure,array([index,i]).reshape(-1,2),axis=0)
#     # print(index,i)
#     centroid_trunk = append(centroid_trunk, candidate_trunk, axis=1)
#     centroid_pole = append(centroid_pole, candidate_pole, axis=1)
#     size_trunk = append(size_trunk, size_trunk[-1]+candidate_trunk.shape[1]).astype(int)
#     size_pole = append(size_pole, size_pole[-1]+candidate_pole.shape[1]).astype(int)
# pose_x_list = []
# pose_y_list = []
# pose_z_list = []
# current_x = []
# current_y = []
# current_z = []
# loop_x = []
# loop_y = []
# loop_z = []
# for i in range(4540):
#     pose_x_list.append(Kitti.odometry.poses[i][0][3])
#     pose_y_list.append(Kitti.odometry.poses[i][1][3])
#     pose_z_list.append(Kitti.odometry.poses[i][2][3])
# for j in range(loop_closure.shape[0]):
#     current_x.append(Kitti.odometry.poses[loop_closure[j][1]][0][3])
#     current_y.append(Kitti.odometry.poses[loop_closure[j][1]][1][3])
#     current_z.append(Kitti.odometry.poses[loop_closure[j][1]][2][3])
#     loop_x.append(Kitti.odometry.poses[loop_closure[j][0]][0][3])
#     loop_y.append(Kitti.odometry.poses[loop_closure[j][0]][1][3])
#     loop_z.append(Kitti.odometry.poses[loop_closure[j][0]][2][3])
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(pose_x_list,pose_z_list,s=1,c="b")
# ax.scatter(current_x,current_z,s=2,c="g")
# ax.scatter(loop_x,loop_z,s=2,c="r")
# plt.show()
end_time = time.time()
print("用时: ",end_time-start)
# save("./loop_closure00.npy", loop_closure)
# savetxt("semantic_kind_car.txt",icp_res.reshape(-1,3),fmt="%d")

np.savetxt("./weight08.txt",icp_res.reshape(-1,4),fmt="%.4f")




