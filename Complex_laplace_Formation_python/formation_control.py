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
W = np.zeros((n, n), dtype=np.complex128)
for i in range(n):
    NBR = SrchNbr(i+1, edge)    # Convert to 1-based index
    Wi = compute_weight_i(NBR, r_complex, i+1, n)
    W[i] = Wi
    sum_i = 0
    for j in range(n):
        sum_i -= Wi[j]
    W[i, i] = sum_i

# examinate the result
# print(np.dot(W, r_complex))   # 理论上 Lp=0，这里Wr_0 应该也等于0

# choose leaders: choose the last two agents as leaders
Wf = W[0:n-2, :]
Wfl = W[0:n-2, n-2:]
Wff = W[0:n-2, 0:n-2]

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
v0 = np.zeros((n, 1), dtype=complex)  # 初始速度
x_t = x0    # t时刻，agents 位置复数表示
v_t = v0    # t时刻，agents 速度复数表示

Vec_xL_t_0 = r[n-2:]    # 领导者初始位置，向量表示
Vec_vL_t_0 = np.array([[0, 0], [0, 0]])  # 领导者初始速度，向量表示

# 基本参数设定
dt = 0.5
loop = 0
aL = 1  # 领导者控制参数
aF = 0.4  # 跟随者控制参数

# 误差记录
err_track = np.zeros((n, 1))
err_all = np.zeros((n, total_frames))
x_all = np.zeros((n, total_frames), dtype=np.complex128)

# 主循环
def update(frame):
    global loop, x_t, v_t, ps_t, Vec_xL_t_0
    if loop >= qr.shape[0]:
        return

    # t时刻，领导者目标位置
    A = qr[loop, :4].reshape(2, 2)      # 旋转、放缩矩阵
    b = qr[loop, 4:6]                   # 平移矩阵
    Vec_xL_t = Vec_xL_t_0 @ A.T + b     # t时刻，leaders 目标位置的向量表示
    xL_target = Vec_xL_t[:, 0] + 1j * Vec_xL_t[:, 1]    # t时刻，leaders 目标位置的复数表示
    xF_target = -np.linalg.inv(Wff) @ Wfl @ xL_target   # t时刻，followers 目标位置的复数表示
    x_target = np.concatenate((xF_target, xL_target))   # t时刻，所有 agents 目标位置的复数表示

    # 领导者速度更新
    for i in range(n - 2, n):
        v_t[i] = -aL * (np.tanh(np.real(x_t[i] - xL_target[i - (n - 2)])) +
                        1j * np.tanh(np.imag(x_t[i] - xL_target[i - (n - 2)])))

    # 跟随者速度更新 
    v_t[:n - 2] = (-aF * (x_t[:n - 2] + np.linalg.inv(Wff) @ Wfl @ x_t[n - 2:n])).reshape(n-2,1) - \
                  np.linalg.inv(Wff) @ Wfl @ v_t[n - 2:n]

    # 位置更新
    x_t += (v_t * dt).flatten()

    # 记录数据
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
    for i in range(edge.shape[0]):
        start_point = x_t[edge[i, 0] - 1]
        end_point = x_t[edge[i, 1] - 1]
        plt.plot([np.real(start_point), np.real(end_point)], [np.imag(start_point), np.imag(end_point)], 'k', linewidth=1)
    ax.plot(np.real(x_t[:n-2]), np.imag(x_t[:n-2]), 'k.', markersize=5)
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

plot_multiple_curves(err_all, ["Agent 1", "Agent 2", "Agent 3", "Agent 4", "Agent 5"], "Error of each agent", "Time", "Error") 