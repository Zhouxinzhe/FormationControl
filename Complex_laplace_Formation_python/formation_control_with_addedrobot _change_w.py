import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from self_functions import *
from matplotlib.animation import FuncAnimation

"""
    初始化：
    1. 给定编队图结构，关联矩阵 D: row:5 agents, colomn:7 edges
    2. 给定编队各个机器人的初始位置：
        坐标矩阵 r: the initial positions of the 5 agents
        复数列表 r_complex: the initial positions of the agents in complex number
    3. 给定运动空间维度 d: dimension
"""
# 定义关联矩阵 D: D is the incidence matrix, row:5 agents, colomn:7 edges
D = np.array([
    [1, 1, 1, 0, 0, 0, 0],
    [-1, 0, 0, 1, 1, 0, 0],
    [0, -1, 0, -1, 0, 1, 1],
    [0, 0, -1, 0, 0, -1, 0],
    [0, 0, 0, 0, -1, 0, -1]
])

# 定义 r 矩阵: r is the initial positions of the agents
r = np.array([
    [0.5, 0.5],
    [0.5, -0.5],
    [0, 0],
    [-1, 1],
    [-1, -1]
])

# 定义 r_complex 矩阵: r_complex is the initial positions of the agents in complex number
r_complex = np.array([0.5+0.5j,
       0.5-0.5j,
       0  +  0j,
       -1 +  1j,
       -1 -  1j
])

# 维度 d: dimension is 2, agents moves in 2-dimensional world
d = 2


"""
    根据初始化信息，求取相关信息：
    1. 根据关联矩阵 D，获取 agents 个数与 edges 条数
    2. 根据关联矩阵 D 计算边集 edge (节点索引从 1 开始)
    3. 绘制初始化状态
"""
# n is num of agents, m is num of edges
n, m = D.shape

# 通过关联矩阵 D 求取边集 edge
non_zero_indices_row, non_zero_indices_col = np.where(D != 0)       # 找到 D 中非零元素的索引
non_zero_indices = sorted(non_zero_indices_row + non_zero_indices_col * n + 1)      # 计算 non_zero_indices
edge = np.mod(np.reshape(non_zero_indices, (m, 2)), n)      # 将索引转换为 Mx2 的矩阵
edge[edge == 0] = n     # 将 0 替换为 n

# 绘制初始编队状态
draw_current_state(r, edge)




"""
    控制核心：
    1. 构建权重矩阵（负复拉普拉斯矩阵，W = -L）
"""
# Construct the weight matrix W = -L
W, Wf, Wfl, Wff = generate_complex_weight_matrix(edge, r_complex)

# verify the position of followers which is determined by the chosen leaders
rL = r_complex[n-2:]
rF = -np.linalg.inv(Wff) @ Wfl @ rL
# print("\nPositions rL:")
# print(rL)
# print("\nPositions rF:")
# print(rF)




"""
    运动轨迹 trajectory 设计（相当于自己的数据集）
"""
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

# Print results
# print("Generated trajectory qr:")
# print(qr)
# print("Time sequence tr:")
# print(tr)





"""
    Control
"""
# 初始化
x0 = r_complex  # 初始位置
v0 = np.zeros((n,), dtype=complex)  # 初始速度
x_t = x0    # t时刻，agents 位置复数表示
v_t = v0    # t时刻，agents 速度复数表示

Vec_xL_t_0 = r[n-2:]    # 领导者初始位置，向量表示
Vec_vL_t_0 = np.array([[0, 0], [0, 0]])  # 领导者初始速度，向量表示

# 基本参数设定
dt = 0.5
loop = 0
aL = 1  # 领导者控制参数
aF = 0.4  # 跟随者控制参数

# 要加入的机器人的参数设定
L = 0.5                 # 加入编队后与相邻两个机器人的间距
thres_add = 0.6        # 加入编队的阈值，与目标位置距离小于该值时，认为到达指定位置，加入编队
getT = 0                # 是否到达目标位置的标志
Wchanged = 0            # 是否改变了权重矩阵的标志 (机器人加入编队后，需要改变权重矩阵)
a_s = 3                 # 要加入的机器人的控制参数
thres_us = 0.5          # 要加入的机器人的速度阈值
ps = np.complex128(10 + 3j)            # 要加入的机器人的初始位置, 复数表示
ps_t = ps               # t时刻，要加入的机器人的位置复数表示
tarPos = 0 + 0j         # 要加入的机器人的目标位置
nbr1 = 0                # 要加入的机器人的邻居1，index 1-based
nbr2 = 0                # 要加入的机器人的邻居2，index 1-based

# 误差记录
err_track = np.zeros((n+1, 1))
err_all = np.zeros((n+1, total_frames))
x_all = np.zeros((n+1, total_frames), dtype=np.complex128)
# err_atk_all = np.zeros((1, 1000))
# v_all = np.zeros((5, 1000))
# ps_all = np.zeros((1, 1000))


# 主循环
def update(frame):
    global loop, x_t, v_t, ps_t, Vec_xL_t_0
    global getT, Wchanged, a_s, thres_us, thres_add
    global tarPos, nbr1, nbr2
    global W, Wf, Wfl, Wff, n, edge
    if loop >= qr.shape[0]:
        return

    # t时刻，领导者目标位置
    A = qr[loop, :4].reshape(2, 2)      # 旋转、放缩矩阵
    b = qr[loop, 4:6]                   # 平移矩阵
    Vec_xL_t = Vec_xL_t_0 @ A.T + b     # t时刻，leaders 目标位置的向量表示
    xL_target = Vec_xL_t[:, 0] + 1j * Vec_xL_t[:, 1]    # t时刻，leaders 目标位置的复数表示
    xF_target = -np.linalg.inv(Wff) @ Wfl @ xL_target   # t时刻，followers 目标位置的复数表示
    x_target = np.concatenate((xF_target, xL_target))   # t时刻，所有 agents 目标位置的复数表示

    # 要加入的机器人的速度更新
    if getT == 0:
        tarPos, nbr1, nbr2 = FindTargetPostion(x_t, ps_t, edge, L)    # 要加入的机器人的目标位置
        us = -a_s * (np.tanh(np.real(ps_t - tarPos)) + 1j * np.tanh(np.imag(ps_t - tarPos)))    # 要加入的机器人的速度
        # limit the speed
        if np.abs(us) > thres_us:
            us = thres_us * us / np.abs(us)
        # 判断是否到达目标位置
        if np.abs(ps_t - tarPos) < thres_add:
            getT = 1
            # 将要加入的机器人的信息加入到原本的编队中
            n += 1
            x_t = np.insert(x_t, 0, ps_t)
            v_t = np.insert(v_t, 0, us)
            edge = update_edge(edge, nbr1, nbr2)
            W, Wf, Wfl, Wff = generate_complex_weight_matrix(edge, x_t)
            xF_target = -np.linalg.inv(Wff) @ Wfl @ xL_target
            x_target = np.concatenate((xF_target, xL_target))
            Wchanged = 1

    # 领导者速度更新
    for i in range(n - 2, n):
        v_t[i] = -aL * (np.tanh(np.real(x_t[i] - xL_target[i - (n - 2)])) +
                        1j * np.tanh(np.imag(x_t[i] - xL_target[i - (n - 2)])))

    # 跟随者速度更新 
    v_t[:n - 2] = (-aF * (x_t[:n - 2] + np.linalg.inv(Wff) @ Wfl @ x_t[n - 2:n])) - \
                  (np.linalg.inv(Wff) @ Wfl @ v_t[n - 2:n])

    # 位置更新
    x_t += v_t * dt

    if Wchanged == 0:
        ps_t += us * dt

    # 记录数据
    if Wchanged == 1:
        err_track = x_t - x_target
        err_all[:, loop] = np.abs(err_track)
        x_all[:, loop] = x_t
    # err_atk = x_atk - x_t[1]
    # err_atk_all[:, loop - 1] = np.real(err_atk)
    # v_all[:, loop - 1] = np.real(v_t[:5].flatten())
    # ps_all[:, loop - 1] = np.real(x_atk)

    # 可视化
    ax.clear()

    # current position
    # plot edges
    for i in range(edge.shape[0]):
        start_point = x_t[edge[i, 0] - 1]
        end_point = x_t[edge[i, 1] - 1]
        if Wchanged == 1 and (edge[i, 0] == 1 or edge[i, 1] == 1):
            plt.plot([np.real(start_point), np.real(end_point)], [np.imag(start_point), np.imag(end_point)], 'g', linewidth=1)
        else:
            plt.plot([np.real(start_point), np.real(end_point)], [np.imag(start_point), np.imag(end_point)], 'k', linewidth=1)
    # if getT == 1:
    #     plt.plot([np.real(ps_t), np.real(x_t[nbr1-1])], [np.imag(ps_t), np.imag(x_t[nbr1-1])], 'g', linewidth=1)
    #     plt.plot([np.real(ps_t), np.real(x_t[nbr2-1])], [np.imag(ps_t), np.imag(x_t[nbr2-1])], 'y', linewidth=1)
    
    # plot nodes
    if Wchanged == 0:
        ax.plot(np.real(x_t[:n-2]), np.imag(x_t[:n-2]), 'k.', markersize=5)
        ax.plot(np.real(x_t[n-2:]), np.imag(x_t[n-2:]), 'b.', markersize=5)
        ax.plot(np.real(ps_t), np.imag(ps_t), 'r.', markersize=5)
    else:
        ax.plot(np.real(x_t[0]), np.imag(x_t[0]), 'r.', markersize=5)
        ax.plot(np.real(x_t[1:n-2]), np.imag(x_t[1:n-2]), 'k.', markersize=5)
        ax.plot(np.real(x_t[n-2:]), np.imag(x_t[n-2:]), 'b.', markersize=5)
    
    # target position
    # ax.plot(np.real(x_target), np.imag(x_target), 'g.', markersize=5)

    ax.set_xlim(-5, 15)
    ax.set_ylim(-25, 5)
    ax.set_title(f"Frame {loop}")
    ax.grid(True)

    loop += 1
    # print(f"Frame {loop} is done.")

fig, ax = plt.subplots(figsize=(10, 6))
ani = FuncAnimation(fig, update, frames=int(qr.shape[0]), interval=1)
# ani.save("pid_animation.gif", writer="pillow", fps=30)
plt.show()

plot_multiple_curves(err_all, ["New Agent", "Agent 1", "Agent 2", "Agent 3", "Agent 4", "Agent 5"], "Error of each agent", "Time", "Error") 