# Description: Formation control
# wij = [a b; -b a]

import numpy as np
import matplotlib.pyplot as plt
from self_functions import *
from matplotlib.animation import FuncAnimation

# 定义矩阵 D: D is the incidence matrix, row:5 agents, colomn:7 edges
D = np.array([
    [1, 1, 1, 0, 0, 0, 0],
    [-1, 0, 0, 1, 1, 0, 0],
    [0, -1, 0, -1, 0, 1, 1],
    [0, 0, -1, 0, 0, -1, 0],
    [0, 0, 0, 0, -1, 0, -1]
])

# 维度 d: dimension is 2, agents moves in 2-dimensional world
d = 2

# 获取矩阵 D 的大小: n is num of agents, m is num of edges
n, m = D.shape

# 定义 r 矩阵: r is the initial positions of the agents
r = np.array([
    [0.5, 0.5],
    [0.5, -0.5],
    [0, 0],
    [-1, 1],
    [-1, -1]
])

p = r.flatten().reshape(-1, 1)

# 通过邻接矩阵 D 求取边集 edge
"""
D = 
[[ 1  1  1  0  0  0  0]
 [-1  0  0  1  1  0  0]
 [ 0 -1  0 -1  0  1  1]
 [ 0  0 -1  0  0 -1  0]
 [ 0  0  0  0 -1  0 -1]]

"""
# 找到 D 中非零元素的索引
non_zero_indices_row, non_zero_indices_col = np.where(D != 0)
# 计算 non_zero_indices
non_zero_indices = sorted(non_zero_indices_row + non_zero_indices_col * n + 1)
# 将索引转换为 Mx2 的矩阵
edge = np.mod(np.reshape(non_zero_indices, (m, 2)), n)
# 将 0 替换为 n
edge[edge == 0] = n

# 绘图
plt.figure()

# 绘制边（黑色线段）
for i in range(edge.shape[0]):
    start_point = r[edge[i, 0] - 1]
    end_point = r[edge[i, 1] - 1]
    plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'k', linewidth=1)

# 绘制节点（黑色点）
for i in range(r.shape[0]):
    plt.plot(r[i, 0], r[i, 1], 'k.', markersize=30)

# 设置坐标轴范围
plt.axis([-5, 5, -5, 5])

# 显示图形
plt.show()

# construct weights W
W = np.zeros((n*d, n*d))
for i in range(n):
    NBR = SrchNbr(i+1, edge)    # Convert to 1-based index
    Wi = compute_weight_i_matrix(NBR, r, i+1, n, d)
    W[d*i:d*i+d, :] = Wi
    sum_i = np.zeros((d, d))
    for j in range(n):
        sum_i -= Wi[:, d*j:d*j+d]
    W[d*i:d*i+d, d*i:d*i+d] = sum_i

# choose leaders: choose the last two agents as leaders
Wf = W[:d*(n-2), :]
Wfl = W[:d*(n-2), d*(n-2):]
Wff = W[:d*(n-2), :d*(n-2)]

# Wff = np.array([[-0.00000000e+00, -6.00493564e-09, -6.00482145e-09,
#         -6.00485458e-09, -6.00478038e-09, -6.00486436e-09],
#        [-4.73053071e-09,  4.73053071e-09, -4.73041524e-09,
#         -4.73030168e-09, -4.73034689e-09, -4.73029569e-09],
#        [-4.72927134e-09, -4.72928398e-09,  9.45855532e-09,
#         -4.72923304e-09, -4.72930474e-09, -4.72930433e-09],
#        [-5.05488998e-09, -5.05489018e-09, -5.05490861e-09,
#          1.51646888e-08, -5.05485938e-09, -5.05493278e-09],
#        [-4.73055176e-09, -4.73055925e-09, -4.73052259e-09,
#         -4.73054974e-09,  1.89221833e-08, -4.73058655e-09],
#        [-5.07906696e-09, -5.07905244e-09, -5.07894367e-09,
#         -5.07912019e-09, -5.07886915e-09,  2.53950524e-08]])

# Wfl = np.array([[-5.78375310e-09, -6.00478264e-09,  5.38234110e-08,
#         -6.00475016e-09],
#        [-4.73027792e-09,  4.23435985e-08, -4.73027842e-09,
#         -4.50074422e-09],
#        [-4.68353355e-09, -4.72934459e-09,  4.25178895e-08,
#         -4.72932256e-09],
#        [-5.05485869e-09,  4.54568268e-08, -5.05488190e-09,
#         -5.01745294e-09],
#        [-4.66815291e-09, -4.73061662e-09,  4.25127692e-08,
#         -4.73061208e-09],
#        [-5.07896672e-09,  4.50973952e-08, -5.07913175e-09,
#         -4.46520156e-09]])

# Define via points 编队中心的轨迹点
via = np.array([
    [0, 0],
    [5, 0],
    [10, 0],
    [10, -10],
    [10, -20],
    [5, -20],
    [0, -20],
    [0, -10],
    [0, 0]
])

ra = np.zeros((via.shape[0], n, 2))     # 编队中每个节点的轨迹点
qvia = np.zeros((via.shape[0], 6))      # 每个轨迹点对应的旋转、缩放、平移，前四列是旋转、缩放，后两列是平移

for j in range(via.shape[0]):
    if j % 2 != 0:
        if j == 3 or j == 7:
            T1 = np.diag([2, 1])
        else:
            T1 = np.diag([1, 0.5])
    else:
        T1 = np.eye(2)

    T2 = rot2(-np.pi / 2 * np.floor(j / 2))  # Rotate every two steps
    ra[j, :, :] = r @ T2.transpose() @ T1.transpose() + via[j, :]
    T = np.dot(T1, T2)
    qvia[j, :] = np.concatenate((T.flatten(), via[j, :]))

# Generate trajectory
qr,dqr,ddqr,tr = mstraj_(qvia, 0.03, 0.5)
total_frames = qr.shape[0]  # trajectory 插值后总的控制点数

# 初始化
xL_0 = r[n-2:].copy()    # 领导者初始位置，向量表示
# vL_0 = np.array([[0, 0], [0, 0]])  # 领导者初始速度，向量表示
# pL_0 = xL_0.flatten().reshape(-1, 1)  # 领导者初始位置，列向量表示
# uL_0 = vL_0.flatten().reshape(-1, 1)  # 领导者初始速度，列向量表示

x_t = r.copy()     # 初始位置
p_t = x_t.flatten().reshape(-1, 1)  # 初始位置，列向量表示
pF_t = p_t[:d*(n-2), :]
pL_t = p_t[d*(n-2):, :]
v_t = np.zeros((n, 2))  # 初始速度


# 参数设定
dt = 0.5
loop = 0
aL = 1  # 领导者控制参数
aF = 0.4  # 跟随者控制参数

# 跟踪误差
# err_track = np.zeros((5, 1))

# 初始化记录
# err_all = np.zeros((5, 1000))
# x_all = np.zeros((5, 1000))
# err_atk_all = np.zeros((1, 1000))
# v_all = np.zeros((5, 1000))
# ps_all = np.zeros((1, 1000))
Data_uL_t = np.zeros((total_frames, 4, 1))  # 领导者速度
Data_uf_t = np.zeros((total_frames, 6, 1))  # 跟随者速度
Data_pL_t = np.zeros((total_frames, 4, 1))  # 领导者位置
Data_pF_t = np.zeros((total_frames, 6, 1))  # 跟随者位置


# 主循环
def update(frame):
    global loop, x_t, p_t, pL_t, pF_t, v_t
# while loop < qr.shape[0]:
    # t时刻，领导者目标位置
    A = qr[loop, :4].reshape(2, 2)
    b = qr[loop, 4:6]
    xL_target = xL_0 @ A.T + b # 向量表示
    
    pL_target = xL_target.flatten().reshape(-1, 1)
    xF_target = -np.linalg.inv(Wff) @ Wfl @ pL_target

    # 领导者速度更新
    for i in range(n - 2, n):
        v_t[i, :] = [-aL * np.tanh(x_t[i, 0] - xL_target[i - (n - 2), 0]), -aL * np.tanh(x_t[i, 1] - xL_target[i - (n - 2), 1])]
    uL_t = v_t[n-2:].flatten().reshape(-1, 1)

    # 跟随者速度更新
    uf_t = (-aF * (pF_t + np.linalg.inv(Wff) @ Wfl @ pL_t)) - np.linalg.inv(Wff) @ Wfl @ uL_t
    v_t = np.vstack((uf_t, uL_t)).reshape(-1, 2)

    # 位置更新
    x_t += v_t * dt
    pF_t += uf_t * dt
    pL_t += uL_t * dt

    # 记录数据
    # err_track = x_t[:5] - xL_target
    # err_atk = x_atk - x_t[1]
    # err_all[:, loop - 1] = np.real(err_track.flatten())
    # x_all[:, loop - 1] = np.real(x_t[:5].flatten())
    # err_atk_all[:, loop - 1] = np.real(err_atk)
    # v_all[:, loop - 1] = np.real(v_t[:5].flatten())
    # ps_all[:, loop - 1] = np.real(x_atk)
    Data_pF_t[loop, :, :] = pF_t
    Data_pL_t[loop, :, :] = pL_t
    Data_uf_t[loop, :, :] = uf_t
    Data_uL_t[loop, :, :] = uL_t

    # 可视化
    ax.clear()
    ax.plot(x_t[:, 0], x_t[:, 1], 'k.', markersize=5)
    ax.set_xlim(-5, 15)
    ax.set_ylim(-25, 5)
    ax.set_title(f"Frame {loop}")
    ax.grid(True)

    loop += 1

fig, ax = plt.subplots(figsize=(10, 6))
ani = FuncAnimation(fig, update, frames=int(qr.shape[0]), interval=1)

plt.show()

# 保存数据
np.save("Data_pF_t.npy", Data_pF_t)
np.save("Data_pL_t.npy", Data_pL_t)
np.save("Data_uf_t.npy", Data_uf_t)
np.save("Data_uL_t.npy", Data_uL_t)