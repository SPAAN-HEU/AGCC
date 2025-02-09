from scipy.spatial import KDTree
import yaml
from numpy import *
import open3d as o3d 
# from libs.load_data_360 import *
from scipy.linalg import logm
from libs.global_resource import *
from libs.load_data import *
# from global_optimization_funcV2 import *
# import gtsam
def get_labels_from_yaml(path):
    with open(path,encoding="utf-8") as f:
        content = f.read()
        f.close()
    data = yaml.load(content,yaml.FullLoader).get("labels")
    data = dict(zip(data.values(),data.keys()))
    return data
def same_entity(points):
    dist = square(points - points[:,0].reshape(-1,1))
    dist = sqrt(dist[0,:] ** 2 + dist[1,:] ** 2 + dist[2,:] ** 2)
    trunk_entity = points[:,argwhere(dist < 10).reshape(-1)]
    points = delete(points, argwhere(dist < 10).reshape(-1),axis=1)
    return trunk_entity, points
def static_centroid(pointcloud:ndarray,entity_label:int,semantic_labels:ndarray,size_list:ndarray,centroid_list:ndarray,T_imu_velo,T_w_imu):
    centroid = array([]).reshape(4,-1)
    entity_cloud = pointcloud[:,argwhere(semantic_labels == entity_label).reshape(-1)]
    while entity_cloud.shape[1] > 0:
        entity,entity_cloud= same_entity(entity_cloud)
        centroid = append(centroid,mean(entity,axis=1).reshape(4,-1),axis=1)
    centroid = dot(T_w_imu, dot(T_imu_velo, centroid))
    centroid_list = append(centroid_list, centroid,axis=1)
    size_list = append(size_list, centroid_list.shape[1])
    return centroid_list, size_list.astype(int)
def static_centroidV2(pointcloud:ndarray,entity_label:int,semantic_labels:ndarray,size_list:ndarray,centroid_list:ndarray,T_imu_velo,T_w_imu,i):
    """
    适用于loop_closureV2程序
    """
    centroid = array([]).reshape(4,-1)
    entity_cloud = pointcloud[:,argwhere(semantic_labels == entity_label).reshape(-1)]
    while entity_cloud.shape[1] > 0:
        entity,entity_cloud= same_entity(entity_cloud)
        centroid = append(centroid,mean(entity,axis=1).reshape(4,-1),axis=1)
    if centroid.shape[1] == 0:
        print("no entity",i,entity_label)
        entity = zeros((4,1))
        centroid = append(centroid,entity,axis=1)
    dist = linalg.norm(centroid,axis = 0)
    centroid_candidate = dot(T_w_imu, dot(T_imu_velo, centroid[:,argmin(dist)])).reshape(-1,1)
    centroid_list = append(centroid_list,centroid_candidate,axis=1)
    return centroid_list,centroid.shape[1]
    
def dynamic_centroidV3(instance_labels:ndarray,_centroid:ndarray,pointcloud:ndarray,semantic_labels:ndarray,semantic_label:int,T_imu_vleo,T_w_imu):
    """
    适用于loop_closureV3程序
    """
    _label = set(instance_labels[argwhere(semantic_labels == semantic_label).reshape(-1)])
    _centroid_list = array([]).reshape(4,-1)
    if len(_label) == 0:
        _centroid = append(_centroid, zeros((4, 1)), axis=1)
        print("no entity")
        return _centroid
    for i in _label:
        _centre = mean(pointcloud[:,argwhere(instance_labels == i).reshape(-1)], axis=1).reshape(4,-1)
        _centroid_list = append(_centroid_list,_centre, axis=1)
    dist = linalg.norm(_centroid_list,axis = 0)
    _centroid_candidate = dot(T_w_imu,dot(T_imu_vleo,_centroid_list[:,argmin(dist).reshape(-1)])).reshape(-1,1)
    _centroid = append(_centroid, _centroid_candidate, axis=1)
    return _centroid
def dynamic_centroid(instance_labels:ndarray,_centroid:ndarray,pointcloud:ndarray,semantic_labels:ndarray,semantic_label:int,T_imu_vleo,T_w_imu):
    _label = set(instance_labels[argwhere(semantic_labels == semantic_label).reshape(-1)])
    for i in _label:
        _centre = mean(pointcloud[:,argwhere(instance_labels == i).reshape(-1)], axis=1).reshape(4,-1)
        _centroid = append(_centroid,_centre, axis=1)
    _size = append(_size, _label.__len__())
    _centroid = dot(T_w_imu,dot(T_imu_vleo,_centroid))
    return _centroid,_size
def loop_centroid(pointcloud:ndarray,semantic_labels:ndarray,entity_label:int,T_imu_velo:ndarray,T_w_imu:ndarray)->ndarray:
    centroid = array([]).reshape(4,-1)
    entity_cloud = pointcloud[:,argwhere(semantic_labels == entity_label).reshape(-1)]
    while entity_cloud.shape[1] > 0:
        entity,entity_cloud= same_entity(entity_cloud)
        centroid = append(centroid, mean(entity, axis=1).reshape(4,-1),axis=1)
    centroid = dot(T_w_imu, dot(T_imu_velo, centroid))
    return centroid

def loop_centroidV2(pointcloud:ndarray,semantic_labels:ndarray,entity_label:int,T_imu_velo:ndarray,T_w_imu:ndarray,i)->ndarray:
    """
    适用于loop_closureV2
    """
    centroid = array([]).reshape(4,-1)
    entity_cloud = pointcloud[:,argwhere(semantic_labels == entity_label).reshape(-1)]
    while entity_cloud.shape[1] > 0:
        entity,entity_cloud= same_entity(entity_cloud)
        centroid = append(centroid, mean(entity, axis=1).reshape(4,-1),axis=1)
    if centroid.shape[1] == 0:
        print("no entity",i,entity_label)
        entity = zeros((4,1))
        centroid = append(centroid,entity,axis=1)
    dist = linalg.norm(centroid,axis=0)
    centroid_candidate = dot(T_w_imu, dot(T_imu_velo, centroid[:,argmin(dist)]))
    return centroid_candidate,centroid.shape[1]
def loop_centroid_car(instance_labels:ndarray,pointcloud:ndarray,semantic_labels:ndarray,semantic_label:int,T_imu_vleo,T_w_imu):
    _label = set(instance_labels[argwhere(semantic_labels == semantic_label).reshape(-1)])
    _centroid_list = array([]).reshape(4,-1)
    if len(_label) == 0:
        _centroid_candidate = array([999,999,999,999])
        return _centroid_candidate
    for i in _label:
        _centre = mean(pointcloud[:,argwhere(instance_labels == i).reshape(-1)], axis=1).reshape(4,-1)
        _centroid_list = append(_centroid_list,_centre, axis=1)
    dist = linalg.norm(_centroid_list,axis = 0)
    _centroid_candidate = dot(T_w_imu,dot(T_imu_vleo,_centroid_list[:,argmin(dist)]))
    return _centroid_candidate
def candidate_frame(index:ndarray,size_entity:ndarray):
    #确定当前帧中的实体与之前哪一帧的距离最近
    if min(index)>= size_entity[-2]:
        frame = full(len(index),size_entity.shape[0]-1)
    else:
        frame = zeros(len(index))
        for i in range(len(index)):
            frame[i] = argmax(size_entity > index[i])
    return frame
def find_loop_candidate(frame_pole:ndarray,frame_trunk:ndarray):
    _frame_array = repeat(frame_pole.reshape(-1,1),frame_trunk.size,axis=1)
    _frame_error = abs(_frame_array - frame_trunk)
    _row_index = argmin(_frame_error, axis=0)
    _col_index = arange(0,frame_trunk.size)
    _candidate_error =  _frame_error[(_row_index,_col_index)]
    _candidate_index = argwhere(_candidate_error <= 20).reshape(-1)
    _frame_trunk = frame_trunk[_candidate_index]
    _frame_pole = frame_pole[_row_index[_candidate_index]]
    return _frame_trunk, _frame_pole
def distance_compute(pos_now,pos_candidate):
    # pos_now = raw.get_pos(frame_now)
    # pos_candidate = raw.get_pos(frame_candidate)
    distance = linalg.norm(pos_now - pos_candidate)
    return distance
def score_compute(centroid_pole_now,centroid_trunk_now,index,size_pole,size_trunk,centroid_trunk,centroid_pole):
    if index != 0:
        _candidate_centroid_trunk = centroid_trunk[:,size_trunk[index-1]:size_trunk[index]]
        _candidate_centroid_pole = centroid_pole[:,size_pole[index-1]:size_pole[index]]
    else:
        _candidate_centroid_trunk = centroid_trunk
        _candidate_centroid_pole = centroid_pole
    _tree_pole = KDTree(_candidate_centroid_pole.T[:,0:3])
    _tree_trunk = KDTree(_candidate_centroid_trunk.T[:, 0:3])
    _distance_pole, _index_pole = _tree_pole.query(centroid_pole_now.T[:, 0:3])
    _distance_trunk, _index_trunk = _tree_trunk.query(centroid_trunk_now.T[:, 0:3])
    score = 0.5*(sum(_distance_pole)/_index_pole.size + sum(_distance_trunk)/_index_trunk.size)
    return score
def entity_extract(semantic_label:int,point_cloud:ndarray,semantic_labels:ndarray):
    entity = array([]).reshape(4,-1)
    entity_size = array([])
    centroid = array([]).reshape(4, -1)
    points = point_cloud[:,argwhere(semantic_labels == semantic_label).reshape(-1)]
    while points.shape[1] > 0:
        dist = linalg.norm(points-points[:,0].reshape(4,-1),axis=0)
        entity_points = points[:,argwhere(dist < 10).reshape(-1)]
        points = delete(points, argwhere(dist < 10).reshape(-1),axis=1)
        entity = append(entity,entity_points, axis=1)
        entity_size = append(entity_size,entity_points.shape[1])
        centroid = append(centroid, mean(entity_points, axis=1).reshape(4, -1), axis=1)
    dist = linalg.norm(centroid,axis=0)
    # print("dist shape: ",dist.shape)
    if dist.shape[0] == 0:
        return nan
    _index = argmin(dist)
    # print("entity size: ",entity_size,_index)
    if _index == 0:
        entity_set = entity[:,0:int(entity_size[0])]
    else:
        entity_set = entity[:,int(sum(entity_size[0:_index])):int(sum(entity_size[0:_index]))+int(entity_size[_index])]
    # print(entity_set.shape)
    return entity_set


        
# def ICP_data(current_frame:int,loop_frame:int,loop_kind:int,database:KittiData,semantic_labels_current:ndarray,semantic_labels_loop:ndarray):
#     current_data = database.get_velo_data(current_frame)
#     current_data = insert(current_data,3, 1, axis=1).T
#     loop_data = database.get_velo_data(loop_frame)  
#     loop_data = insert(loop_data,3, 1, axis=1).T
#     if loop_kind == 0:
#         loop = entity_extract(80, loop_data, semantic_labels_loop)
#         current = entity_extract(80,current_data,semantic_labels_current)
#     if loop_kind == 1:
#         current = entity_extract(71,current_data,semantic_labels_current)
#         loop = entity_extract(71, loop_data, semantic_labels_loop)
#     if loop_kind == 2:
#         current_1 = entity_extract(80, current_data, semantic_labels_current)
#         current_2 = entity_extract(71, current_data, semantic_labels_current)
#         loop_1 = entity_extract(80, loop_data, semantic_labels_loop)
#         loop_2 = entity_extract(71, loop_data, semantic_labels_loop)
#         current = append(current_1, current_2, axis=1)
#         loop = append(loop_1, loop_2, axis=1)
#     if isnan(current).any() or isnan(loop).any():
#         return nan,nan
#     return current, loop
def normal_estimate(pointcloud:ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = asarray(pcd.normals)
    # print(current_pcd.normals[0])
    return normals
def transform_to_se3(T):
    """
    将变换矩阵 T 转换为李代数 se(3)。
    参数:
    T (array): 4x4 的变换矩阵。
    返回:
    array: 6x1 的李代数向量。
    """
    R = T[:3, :3]  # 提取旋转矩阵
    p = T[:3, 3]   # 提取平移向量
    
    theta = arccos((trace(R)-1)/2)
    print("theta: ", theta)
    R_I = R-eye(3)
    U,sigma,Vt = linalg.svd(R_I)
    a = Vt[-1].reshape(1,-1)
    a = a/linalg.norm(a)
    J = sin(theta)/theta*eye(3)+(1-sin(theta)/theta)*a.T.dot(a)+((1-cos(theta))/theta)*array([[0,-a[0,2],a[0,1]],[a[0,2],0,-a[0,0]],[-a[0,1],a[0,0],0]])
    rho = linalg.inv(J).dot(p.reshape(-1,1))
    phi = (theta*a).reshape(-1,1)
    # print("rho: \n", rho.shape)
    # print("phi: ",phi.shape)
    print("sigma: \n",sigma)
    print("R_I: \n",R_I)
    print("a: \n", a)
    print("check: \n",R_I.dot(a.T))
    # 合并 w 和 v 得到李代数向量
    xi = concatenate((rho,phi)).T
    return xi.reshape(-1, 1)
def se3_to_transform(xi):
    theta = linalg.norm(xi[3:6])
    a = (xi[3:6]/theta).reshape(1,-1)
    J = sin(theta)/theta*eye(3)+(1-sin(theta)/theta)*a.T.dot(a)+((1-cos(theta))/theta)*array([[0, -a[0,2], a[0,1]], [a[0,2], 0, -a[0,0]], [-a[0,1], a[0,0], 0]])
    R = cos(theta)*eye(3)+(1-cos(theta))*a.T.dot(a)+sin(theta)*array([[0, -a[0,2], a[0,1]], [a[0,2], 0, -a[0,0]], [-a[0,1], a[0,0], 0]])
    t = J.dot(xi[0:3])
    T = eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
# def sigma_normal(sigma_x,sigma_y,sigma_z):
#     sigma_x_normal = (sigma_x-min(sigma_x))/(max(sigma_x)-min(sigma_x))
#     sigma_y_normal = (sigma_y-min(sigma_y))/(max(sigma_y)-min(sigma_y))
#     sigma_z_normal = (sigma_z-min(sigma_z))/(max(sigma_z)-min(sigma_z))
#     return sigma_x_normal,sigma_y_normal,sigma_z_normal
 
def ICP_weight(current_data:ndarray,loop_data:ndarray,sigma_x:ndarray,sigma_y:ndarray,sigma_z:ndarray,trans_matrix:ndarray):
    jacobin = zeros((6, 6))
    g = zeros((6, 1))
    initial_point = current_data
    xi = zeros((6, 1))
    sigma_x_normal = sigma_x
    sigma_y_normal = sigma_y
    sigma_z_normal = sigma_z
    for k in range(current_data.shape[0]):
        jacobin_j = (-1*array([[1,0,0,0,initial_point[k,2],-initial_point[k,1]],
                               [0,1,0,-initial_point[k,2],0,initial_point[k,0]],
                               [0,0,1,initial_point[k,1],-initial_point[k,0],0]]))
        weight_matrix = diag([sigma_x_normal[k], sigma_y_normal[k], sigma_z_normal[k]])
        # weight_matrix = diag([1, 1, 1])
        jacobin += jacobin_j.T.dot(weight_matrix).dot(jacobin_j)
        f_j = loop_data[k,0:3].T - (trans_matrix.dot(current_data[k].T))[0:3]
        f_j = -jacobin_j.T.dot(weight_matrix).dot(f_j.reshape(-1,1))
        g += f_j
    delta_xi = linalg.inv(jacobin).dot(g) 
    xi += delta_xi
    #xi转T阵不对，应该按照公式展开而不是直接反对称矩阵
    xi = xi.reshape(-1)
    trans_matrix = se3_to_transform(xi)
    # print("trans_matrix: \n",trans_matrix)
    transform_point = trans_matrix.dot(initial_point.T).T
    #这里写一个计算误差RMS的函数，如果RMS小于某一值，那么认为可以停止迭代了，要不然就不断重复
    return trans_matrix
def varience_x(x):
    return 1.81308939e-07*x**2-3.53534438e-06*x+2.02666768e-05
def varience_y(y):
    return 8.21606873e-07*y**2-2.57890720e-05*y+5.16074381e-04
def varience_z(z):
    return 8.21329841e-09*z**2-2.06494095e-07*z+2.92697830e-06

def sigma_estimate(current_icp,loop_icp):
    current_normals = normal_estimate(current_icp[:,0:3])
    loop_normals = normal_estimate(loop_icp[:,0:3])
    #计算每个点的入射角并计算各方向的误差方差
    incidence_current = rad2deg(arccos(sum(current_normals*current_icp[:,0:3],axis=1)/(linalg.norm(current_normals,axis=1)*linalg.norm(current_icp[:,0:3],axis=1))))
    incidence_loop = rad2deg(arccos(sum(loop_normals*loop_icp[:,0:3], axis=1)/(linalg.norm(loop_normals, axis=1)*linalg.norm(loop_icp[:,0:3], axis=1))))
    out_range_current_index = argwhere(incidence_current > 90).reshape(-1)
    out_range_loop_index = argwhere(incidence_loop > 90).reshape(-1)
    incidence_current[out_range_current_index] = abs(180-incidence_current[out_range_current_index])
    incidence_loop[out_range_loop_index] = abs(180-incidence_loop[out_range_loop_index])
    current_x_variance = varience_x(incidence_current)
    loop_x_variance = varience_x(incidence_loop)
    current_y_variance = varience_y(incidence_current)
    loop_y_variance = varience_y(incidence_loop)
    current_z_variance = varience_z(incidence_current)
    loop_z_variance = varience_z(incidence_loop)
    sigma_x = current_x_variance + loop_x_variance
    sigma_y = current_y_variance + loop_y_variance
    sigma_z = current_z_variance + loop_z_variance
    return sigma_x,sigma_y,sigma_z
def sigma_estimate2(current_icp,loop_icp,trans):
    current_normals = normal_estimate(current_icp[:,0:3])
    loop_normals = normal_estimate(loop_icp[:,0:3])
    #计算每个点的入射角并计算各方向的误差方差
    incidence_current = rad2deg(arccos(sum(current_normals*current_icp[:,0:3],axis=1)/(linalg.norm(current_normals,axis=1)*linalg.norm(current_icp[:,0:3],axis=1))))
    incidence_loop = rad2deg(arccos(sum(loop_normals*loop_icp[:,0:3], axis=1)/(linalg.norm(loop_normals, axis=1)*linalg.norm(loop_icp[:,0:3], axis=1))))
    out_range_current_index = argwhere(incidence_current > 90).reshape(-1)
    out_range_loop_index = argwhere(incidence_loop > 90).reshape(-1)
    incidence_current[out_range_current_index] = abs(180-incidence_current[out_range_current_index])
    incidence_loop[out_range_loop_index] = abs(180-incidence_loop[out_range_loop_index])
    current_x_variance = varience_x(incidence_current)
    loop_x_variance = varience_x(incidence_loop)
    current_y_variance = varience_y(incidence_current)
    loop_y_variance = varience_y(incidence_loop)
    current_z_variance = varience_z(incidence_current)
    loop_z_variance = varience_z(incidence_loop)
    coefficient = trans*trans
    current_x = coefficient[0,0]*current_x_variance+coefficient[0,1]*current_y_variance+coefficient[0,2]*current_z_variance
    current_y = coefficient[1,0]*current_x_variance+coefficient[1,1]*current_y_variance+coefficient[1,2]*current_z_variance
    current_z = coefficient[2,0]*current_x_variance+coefficient[2,1]*current_y_variance+coefficient[2,2]*current_z_variance
    sigma_x = current_x + loop_x_variance
    sigma_y = current_y + loop_y_variance
    sigma_z = current_z + loop_z_variance
    # sigma_x_normal = (sigma_x-min(sigma_x))/(max(sigma_x)-min(sigma_x))
    # sigma_y_normal = (sigma_y-min(sigma_y))/(max(sigma_y)-min(sigma_y))
    # sigma_z_normal = (sigma_z-min(sigma_z))/(max(sigma_z)-min(sigma_z))
    # return sigma_x_normal,sigma_y_normal,sigma_z_normal
    return 1/sigma_x,1/sigma_y,1/sigma_z
def position_distance_compute1(current_index,loop_index,Kitti):
    # print(Kitti.raw_data.oxts[current_index])
    lat_1 = deg2rad(Kitti.raw_data.oxts[current_index].packet.lat)
    lon_1 = deg2rad(Kitti.raw_data.oxts[current_index].packet.lon)
    lat_2 = deg2rad(Kitti.raw_data.oxts[loop_index].packet.lat)
    lon_2 = deg2rad(Kitti.raw_data.oxts[loop_index].packet.lon)
    a = abs(lat_1-lat_2)
    b = abs(lon_1-lon_2)
    S = 2*arcsin(sqrt(sin(a/2)**2+cos(lat_1)*cos(lat_2)*sin(b/2)**2))*6378137
    return S
# def position_distance_compute(current_index,loop_index,Kitti:KittiData360):
#     lat_1 = deg2rad(Kitti.raw_data[Kitti.oxts_dir[current_index]].lat)
#     lon_1 = deg2rad(Kitti.raw_data[Kitti.oxts_dir[current_index]].lon)
#     lat_2 = deg2rad(Kitti.raw_data[Kitti.oxts_dir[loop_index]].lat)
#     lon_2 = deg2rad(Kitti.raw_data[Kitti.oxts_dir[loop_index]].lon)
#     a = abs(lat_1-lat_2)
#     b = abs(lon_1-lon_2)
#     S = 2*arcsin(sqrt(sin(a/2)**2+cos(lat_1)*cos(lat_2)*sin(b/2)**2))*6378137
#     return S
def quat_to_pos_matrix_hm(p_x, p_y, p_z, x, y, z, w):
    # 创建位姿矩阵，写入位置
    T = array([[0, 0, 0, p_x], [0, 0, 0, p_y], [0, 0, 0, p_z], [0, 0, 0, 1]])
    T[0, 0] = 1 - 2 * pow(y, 2) - 2 * pow(z, 2)
    T[0, 1] = 2 * (x * y - w * z)
    T[0, 2] = 2 * (x * z + w * y)

    T[1, 0] = 2 * (x * y + w * z)
    T[1, 1] = 1 - 2 * pow(x, 2) - 2 * pow(z, 2)
    T[1, 2] = 2 * (y * z - w * x)

    T[2, 0] = 2 * (x * z - w * y)
    T[2, 1] = 2 * (y * z + w * x)
    T[2, 2] = 1 - 2 * pow(x, 2) - 2 * pow(y, 2)
    return T

# def saveOptimizedGraphPose(curr_node_idx, graph_optimized):

#     for opt_idx in range(curr_node_idx):
#         pose_trans, pose_rot = getGraphNodePose(graph_optimized, opt_idx)
#         pose_trans = reshape(pose_trans, (-1, 3)).squeeze()
#         pose_rot = reshape(pose_rot, (-1, 9)).squeeze()
#         optimized_pose_ith = array([ pose_rot[0], pose_rot[1], pose_rot[2], pose_trans[0], 
#                                         pose_rot[3], pose_rot[4], pose_rot[5], pose_trans[1], 
#                                         pose_rot[6], pose_rot[7], pose_rot[8], pose_trans[2],
#                                         0.0, 0.0, 0.0, 0.1 ])
#     #     if(opt_idx == 0):
#     #         optimized_pose_list = optimized_pose_ith
#     #     else:
#     #         optimized_pose_list = vstack((optimized_pose_list, optimized_pose_ith))
#     # save("./Result/optimized_result.npy",optimized_pose_list)

if __name__ == '__main__':
    Kitti = KittiData(basedir,date,drive,sequence[0])
    print(position_distance_compute1(3803,888,Kitti))
