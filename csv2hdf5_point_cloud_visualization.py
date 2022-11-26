# import pandas as pd
# import ast
# df = pd.read_csv('C:/Users/84168/Desktop/export_data__qznvhu6p1IqR3J42yL71.csv')
#
# dic1 = df.to_dict()
# # print(type(dic1))
# # # print(dic1)
# # print(len(dic1['points']))
# # print(type(dic1['points']))
# # print(dic1['points'][0])
# # print(dic1['points'][0][1:len(dic1['points'][0])-1])
#
# # print(type(dic1['points'][3][1:len(dic1['points'][0])-1]))
# # i = 0
# # range = []
# # azimuth = []
# # elevation = []
# # for x in dic1['points'][3][1:len(dic1['points'][3])-1]:
# #     if x =='{':
# #         i = i + 1
# #
# # print(i)
# # 将字符串类型转换为元组类型
# user_dict = ast.literal_eval(dic1['points'][3][1:len(dic1['points'][3])-1])
# print(user_dict)
# print(type(user_dict))
# print(len(user_dict))
# # 索引元组中的元素，得到单个点的字典类型数据
# print(user_dict[0])
# # print(type(user_dict[0]))
# print(user_dict[0]["range"])
#
# # the number of points
# length = len(dic1['points'])
# print(length)
#
# # for i in range()
# # for t in range(5):
# #     print(t)

import h5py
import pandas as pd
import scipy.io
import ast
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# Parameter settings
pi = 3.1415926
# -------------------------------------------------------------------- #
# To reduce the amount of data transferred with Wi-Fi, the data type of actual transmission is int8_t.
# Therefore, the real data should be multiplied by the following float-type Unit.
# -------------------------------------------------------------------- #
azimuthUnit = 0.01
elevationUnit = 0.01
rangeUnit = 0.00025
dopplerUnit = 0.00028

# ===============================================================================================================
# ----------------------------------------------------------------
# Following are the helper functions to load save/load HDF5 files
# ----------------------------------------------------------------

# Write numpy array data and label to h5_filename
def save_h5_data_label_normal(h5_filename, data, label, normal,
		data_dtype='float32', label_dtype='uint8', normal_dtype='float32'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'normal', data=normal,
            compression='gzip', compression_opts=4,
            dtype=normal_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()


# Write numpy array data and label to h5_filename
def save_h5(h5_filename, data, label, data_dtype='float32', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()

# Read numpy array data and label from h5_filename
def load_h5_data_label_normal(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    normal = f['normal'][:]
    return (data, label, normal)

# Read numpy array data and label from h5_filename
def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)

# Read numpy array data and label from h5_filename
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

# Main process.
# There is a big loop (whose variable is k)whose main function is to process the data per posture one by one.
# =========================================================================================================
# read the .csv format data files.
# df = pd.read_csv('C:/Users/84168/Desktop/export_data_V3ljhJBtZDw5kFg66e0Z8.csv')
# store all the data in one dataset.
# Before running this code, change the following parameter to the number of postures.
dataperdataset = np.zeros((5, 256, 4), dtype = float)

for k in range(5):      # k = the number of postures
    filename = 'D:/下载/multiposture_data_set/hunker/hunker_'+ str(k+1) +'.csv'
    # df = pd.read_csv('D:/下载/multiposture_data_set/bow/bow_2.csv')
    df = pd.read_csv(filename)
    dic1 = df.to_dict()  # transfer the data to dict class


    # Lists used to store the 3D coordinates of points: range, azimuth, elevation and Dopper dimension.
    # Stored in the radian format.
    range_point = []
    azimuth = []
    elevation = []
    doppler = []

    # The total number of frames in the package.
    frame_num = len(dic1['points'])
    # print(frame_num)
    num_pointcloud = 0   # the number of points taken into the process.

    for i in range(80):
        # get the point cloud in one single frame
        total_length = len(dic1['points'][i])
        data_per_frame = dic1['points'][i][1:total_length-1]
        # transfer string-class point cloud in a single frame to tuple class.
        dict_per_frame = ast.literal_eval(dic1['points'][i])
        # print(dict_per_frame)
        # print(len(dict_per_frame))
        point_num = len(dict_per_frame)

        # store the data in the corresponding lists
        for j in range(point_num):
            range_point.append(dict_per_frame[j]["range"]*rangeUnit)
            azimuth.append(dict_per_frame[j]["azimuth"]*azimuthUnit)
            elevation.append(dict_per_frame[j]["elevation"]*elevationUnit)
            doppler.append(dict_per_frame[j]["doppler"]*dopplerUnit)

        num_pointcloud = num_pointcloud + point_num

    # store the data in the single posture sample, which mincludes all the feature of one single posture.
    dataperposture = np.zeros((1, 256, 4), dtype=float)
    if num_pointcloud >= 256:
        for i in range(256):
            dataperposture[0][i][0] = range_point[i]
            dataperposture[0][i][1] = elevation[i]
            dataperposture[0][i][2] = azimuth[i]
            dataperposture[0][i][3] = doppler[i]

    # transfer data from dataperframe
    for m in range(256):
        dataperdataset[k][m][0] = dataperposture[0][m][0]
        dataperdataset[k][m][1] = dataperposture[0][m][1]
        dataperdataset[k][m][2] = dataperposture[0][m][2]
        dataperdataset[k][m][3] = dataperposture[0][m][3]

print(dataperdataset)
print(dataperdataset.shape)


# range_point = np.array(range_point)
# range_point.reshape()






# print(num_pointcloud)
# print(dataperframe)
# print(dataperframe.shape)
# print(type(dataperframe))


# ----------------------------------------------------------------------
# Transfer data from Spherical coordinates to Cartesian coordinates.
x = []
y = []
z = []
index = 4
for i in range(256):
    x.append(dataperdataset[index][i][0]*math.cos(dataperdataset[index][i][1])*math.sin(dataperdataset[index][i][2]))
    y.append(dataperdataset[index][i][0]*math.cos(dataperdataset[index][i][1])*math.cos(dataperdataset[index][i][2]))
    z.append(dataperdataset[index][i][0]*math.sin(dataperdataset[index][i][1]))



# ----------------------------------------------------------------------
# visualize the point clouds
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot3D(x, y, z, 'c.')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()


# # ==================================================================
# # generate data set of the HDF5 format.
# Label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0]
# total_data_set = np.array(total_data_set)
# hdf5_filename = ('/media/jerome/Sumsang/PointNet+训练集生成代码/PointNet/pointnet/fall_train100')
# save_h5(hdf5_filename, total_data_set, Label)








# # def load_h5_data_label_normal(h5_filename):
# m = load_h5('C:/Users/84168/Desktop/fall_train1.h5')
# print(m)


# import os
# import sys
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)
# # from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
# import numpy as np
# import h5py
#
# SAMPLING_BIN = os.path.join(BASE_DIR, 'third_party/mesh_sampling/build/pcsample')
#
# SAMPLING_POINT_NUM = 2048
# SAMPLING_LEAF_SIZE = 0.005
#
# # def save_h5(h5_filename, data, label, data_dtype='float32', label_dtype='uint8'):
# Label = 'Fall'
# hdf5_filename = r'C:User\84168\Desktop\fall_train2'
# save_h5(hdf5_filename, dataperframe, Label)











