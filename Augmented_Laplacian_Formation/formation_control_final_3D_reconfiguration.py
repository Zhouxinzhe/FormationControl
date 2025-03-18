# Description: 3D formation control with reconfiguration
# In this file, leader index is in the front, follower index is in the back
# 这份代码里有很多地方是强行打了补丁, 有些地方可能不太合理, 但是为了实现功能, 暂时先这样了

import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from self_functions import *


# 维度 d: dimension is 3, agents moves in 3-dimensional world
d = 3

# 三维编队--正三面体
edges = np.array([
    [1, 3],
    [1, 4],
    [1, 5],
    [2, 3],
    [2, 4],
    [2, 5],
    [3, 4],
    [3, 5],
    [4, 5]
])

edges = np.array([
    [4, 1],
    [4, 2],
    [4, 3],
    [5, 1],
    [5, 2],
    [5, 3],
    [1, 2],
    [1, 3],
    [2, 3]
])

r = np.array([
    [0.5, math.sqrt(3)/2, 0.05],
    [0.5, -math.sqrt(3)/2, -0.05],
    [-1, 0, 0],
    [0.05, 0, 0.5],
    [-0.05, 0, -0.5]
])

n = r.shape[0]
m = edges.shape[0]

plot_3d_formation(r, edges)


# 旋转轴
l = np.array([0, 0, 1])     

# 生成权重矩阵 W, Wf, Wfl, Wff
W, Wf, Wfl, Wff = generate_weight_matrix_final(r, edges, l)
W_total = W.transpose(0, 2, 1, 3).reshape(3*n, 3*n)
Wf_total = W_total[3*2:, :]
Wfl_total = W_total[3*2:, :3*2]
Wff_total = W_total[3*2:, 3*2:]



# trajectory
traj = [
    # via points    rotation        scale
    ([0, 0, 0],     [l, 0],             1),
    ([5, 0, 0],     [l, 0],             1),
    ([10, 0, 0],    [l, 0],             1),
    # ([10, -10, 0],  [l, -math.pi/2],    0.5),
    # ([10, -20, -5], [l, -math.pi],      0.75),
    # ([5, -20, 0],   [l, -math.pi],      1.25),
    # ([0, -20, 5],   [l, -math.pi],      1),
    # ([0, -10, 2.5], [l, -3*math.pi/2],  1),
    # ([0, 0, 0],     [l, -2*math.pi],    1)
]


# Initialize variables
ra = np.zeros((n, 3, len(traj))) # (agents num, dimension, via points num)
qvia = np.zeros((len(traj), 13))  # (via points num, translation(3) + rotation(9) + scale(3))

for j in range(len(traj)):
    translation = traj[j][0]
    rotation = R.from_rotvec(traj[j][1][1] * np.array(traj[j][1][0])).as_matrix()
    scale = traj[j][2]
    qvia[j, :] = np.concatenate([translation, rotation.flatten(), [scale]])
    

# Generate trajectory
qr,dqr,ddqr,tr = mstraj_(qvia, 0.03, 0.2)



# 初始化
p_0 = r             # 初始位置，向量表示 (7, 3)
pF_0 = r[2:, :]   # 跟随者初始位置，向量表示 (5, 3)
pL_0 = r[:2, :]      # 领导者初始位置，向量表示 (2, 3)
v_0 = np.zeros((n, 3))      # 初始速度，向量表示 (7, 3)
vF_0 = np.zeros((n-2, 3))   # 跟随者初始速度，向量表示 (5, 3)
vL_0 = np.zeros((2, 3))     # 领导者初始速度，向量表示 (2, 3)

p_t = p_0.copy()
pF_t = pF_0.copy()
pL_t = pL_0.copy()
v_t = v_0
vF_t = vF_0
vL_t = vL_0

# 参数设定
t = 0
dt = 0.5
loop = 0
aL = 1  # 领导者控制参数
aF = 1  # 跟随者控制参数

# 要加入的 agent 的参数设定
L = 1                 # 加入编队后与相邻两个 agent 的间距
thres_add = 0.3         # 加入编队的阈值，与目标位置距离小于该值时，认为到达指定位置，加入编队
getT = 0                # 是否到达目标位置的标志
last_getT = 0           # 上一次到达目标位置的标志
aN = 1                  # 要加入的 agent 的控制参数
us = np.array([0.0,0.0,0.0])        # 要加入的 agent 的速度
thres_us = 0.05          # 要加入的 agent 的速度阈值
ps_0 = np.array([12.0, 5, 4.0])  # 要加入的 agent 的初始位置
ps_t = ps_0.copy()      # t时刻，要加入的 agent 的位置
tarPos = np.array([0, 0, 0])         # 要加入的 agent 的目标位置


# 添加轨迹记录和编队帧记录
trajectory_f = []  # 跟随者轨迹
trajectory_l = []  # 领导者轨迹
trajectory_n = []  # 新加入的 agent 的轨迹
formation_frames = []  # 编队帧记录
new_agent_frames = []  # 新加入的 agent 的帧记录

# 用于记录目标位置和实际位置
leader_target_positions = []
leader_actual_positions = []
follower_target_positions = []
follower_actual_positions = []
new_agent_target_positions = []
new_agent_actual_positions = []


# 创建图形和三维轴
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-5, 15)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=5, azim=-90)  # 设置初始视角

# 创建初始的散点图和线条
points_f, = ax.plot([], [], [], 'o', markersize=2, color='blue')  # 初始为空，将在init函数中设置
points_l, = ax.plot([], [], [], 'o', markersize=2, color='red')  # 初始为空，将在init函数中设置
points_n, = ax.plot([], [], [], 'o', markersize=2, color='green')  # 初始为空，将在init函数中设置
lines = []
title_text = ax.text2D(0.5, 0.95, "", transform=ax.transAxes, fontsize=14, ha='center')
for j, k in edges:
    line, = ax.plot([], [], [], color='gray', lw=0.8, alpha=0.6)
    lines.append(line)

# 动画初始化函数
def init():
    # 设置初始位置
    points_f.set_data(p_0[2:,0], p_0[2:,1])
    points_f.set_3d_properties(p_0[2:,2])
    points_l.set_data(p_0[:2,0], p_0[:2,1])
    points_l.set_3d_properties(p_0[:2,2])
    points_n.set_data([ps_0[0]], [ps_0[1]])
    points_n.set_3d_properties([ps_0[2]])
    for i, (j, k) in enumerate(edges):
        if i >= m:
            break
        lines[i].set_data([p_0[j-1, 0], p_0[k-1, 0]], [p_0[j-1, 1], p_0[k-1, 1]])
        lines[i].set_3d_properties([p_0[j-1, 2], p_0[k-1, 2]])
    # title_text = ax.text2D(0.5, 0.95, "t = 0.0", transform=ax.transAxes, fontsize=14, ha='center')
    return (points_f, points_l) + tuple(lines) +  (title_text,)

# 动画更新函数
def update(frame):
    global p_t, pF_t, pL_t, v_t, vF_t, vL_t, loop, t, dt, aL, aF, qr, edges
    global n, edges, getT, ps_t, aN, thres_us, thres_add, L, us, tarPos, last_getT
    global W, Wf, Wfl, Wff, W_total, Wf_total, Wfl_total, Wff_total
    
    # 帧数超过控制点数时退出
    if loop >= qr.shape[0]:
        loop = qr.shape[0] - 1

    # 要加入的 agent 的速度更新
    if getT == 0:
        tarPos, nbr1, nbr2, nbr3 = FindTargetPostion(p_t, ps_t, edges, L)    # 要加入的 agent 的目标位置
        us = [aN * np.tanh(tarPos[0]-ps_t[0]), aN * np.tanh(tarPos[1]-ps_t[1]), aN * np.tanh(tarPos[2]-ps_t[2])]   # 要加入的 agent 的速度
        us = np.array(us)
        # limit the speed
        if np.linalg.norm(us) > thres_us:
            us = thres_us * us / np.abs(us)
        # 判断是否到达目标位置
        if np.linalg.norm(ps_t - tarPos) < thres_add:
            getT = 1
            # 将要加入的 agent 的信息加入到原本的编队中
            n += 1
            edges = update_edge(edges, nbr1, nbr2, nbr3, n)
            p_t = np.vstack((p_t, ps_t))
            W, Wf, Wfl, Wff = generate_weight_matrix_final(p_t, edges, l)
            W_total = W.transpose(0, 2, 1, 3).reshape(3*n, 3*n)
            Wf_total = W_total[3*2:, :]
            Wfl_total = W_total[3*2:, :3*2]
            Wff_total = W_total[3*2:, 3*2:]
            # 新增 lines
            for i in range(3):
                line, = ax.plot([], [], [], color='green', lw=0.8, alpha=0.6)
                lines.append(line)

    
    # loop 时的目标控制
    translation = qr[loop, :3]
    rotation = qr[loop, 3:12].reshape(3, 3)
    scale = qr[loop, 12]

    # 领导者位置的目标位置
    pL_target = scale * rotation @ pL_0.T + translation.reshape(3, 1)   # (3, 2)
    
    # 领导者速度的目标速度
    for i in range(2):
        vL_t[i] = [aL * np.tanh(pL_target[0, i] - pL_t[i, 0]), aL * np.tanh(pL_target[1, i] - pL_t[i, 1]), aL * np.tanh(pL_target[2, i] - pL_t[i, 2])]
    vL_t_total = vL_t.flatten()

    # 跟随者位置的目标位置
    if getT == 0:
        pF_target = (-np.linalg.inv(Wff_total) @ Wfl_total @ (pL_t.flatten().reshape(-1, 1))).reshape(n-2, 3)
    else:
        tmp = (-np.linalg.inv(Wff_total) @ Wfl_total @ (pL_t.flatten().reshape(-1, 1))).reshape(n-2, 3)
        pF_target = tmp[:-1, :]
        tarPos = tmp[-1, :]
    
    # 跟随者位置的目标速度
    if getT == 0:
        vF_t_total = (-aF * (pF_t.flatten() + np.linalg.inv(Wff_total) @ Wfl_total @ pL_t.flatten())) - np.linalg.inv(Wff_total) @ Wfl_total @ vL_t_total
        vF_t = vF_t_total.reshape(n-2, 3)
    else:
        pF_tmp = np.vstack((pF_t, ps_t))
        tmp = (-aF * (pF_tmp.flatten() + np.linalg.inv(Wff_total) @ Wfl_total @ pL_t.flatten())) - np.linalg.inv(Wff_total) @ Wfl_total @ vL_t_total
        tmp = tmp.reshape(n-2, 3)
        vF_t = tmp[:-1, :]
        us = tmp[-1, :]

    # 所有 agent 的位置和速度
    if getT == 0:
        v_t = np.vstack((vL_t, vF_t))
        p_t += v_t * dt
        pL_t = p_t[:2, :]
        pF_t = p_t[2:, :]
        ps_t += us * dt
    else:
        v_t = np.vstack((vL_t, vF_t, us))
        p_t += v_t * dt
        pL_t = p_t[:2, :]
        pF_t = p_t[2:-1, :]
        ps_t = p_t[-1, :]


    # 记录目标位置和实际位置
    leader_target_positions.append(pL_target.T.copy())
    follower_target_positions.append(pF_target.copy())
    leader_actual_positions.append(pL_t.copy())
    follower_actual_positions.append(pF_t.copy())
    new_agent_target_positions.append(tarPos.copy())
    new_agent_actual_positions.append(ps_t.copy())

    # 更新散点位置
    points_f.set_data(pF_t[:,0], pF_t[:,1])
    points_f.set_3d_properties(pF_t[:,2])
    points_l.set_data(pL_t[:,0], pL_t[:,1])
    points_l.set_3d_properties(pL_t[:,2])
    points_n.set_data([ps_t[0]], [ps_t[1]])
    points_n.set_3d_properties([ps_t[2]])
    
    
    # 更新线条位置
    for m, (j, k) in enumerate(edges):
            lines[m].set_data([p_t[j-1, 0], p_t[k-1, 0]], [p_t[j-1, 1], p_t[k-1, 1]])
            lines[m].set_3d_properties([p_t[j-1, 2], p_t[k-1, 2]])

    # 记录轨迹
    trajectory_f.append(pF_t.copy())
    trajectory_l.append(pL_t.copy())
    trajectory_n.append(ps_t.copy())
    
    # 记录编队帧（每隔一定帧数记录一次）
    if loop == 0 or (getT == 1 and last_getT == 0) or loop == qr.shape[0]-2:
        if getT == 0:
            formation_frames.append(p_t.copy())
            new_agent_frames.append(ps_t.copy())
        else:
            formation_frames.append(p_t[:-1,:].copy())
            new_agent_frames.append(ps_t.copy())

    # 更新时间
    t += dt
    loop += 1
    # 更新标题
    title_text.set_text(f"t = {t:.1f}")

    # 更新上一次到达目标位置的标志
    last_getT = getT
    
    return (points_f, points_l) + tuple(lines) + (title_text,)

# 创建动画
ani = animation.FuncAnimation(
    fig, update,
    frames=int(qr.shape[0]),
    init_func=init,
    blit=False,
    interval=1
)
# ani.save("3D_formation_final_reconfiguration.gif", writer="pillow", fps=30)
plt.show()



def plot_trajectories_with_formations():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 2)
    ax.set_yticks([-2, -1, 0, 1, 2])  # 手动指定 y 轴的刻度位置
    ax.set_zlim(-3, 4.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=5, azim=-90)  # 设置初始视角
    
    # 绘制跟随者轨迹
    for i in range(len(trajectory_f[0])):
        x = [traj[i, 0] for traj in trajectory_f]
        y = [traj[i, 1] for traj in trajectory_f]
        z = [traj[i, 2] for traj in trajectory_f]
        ax.plot(x, y, z, color='blue', alpha=0.3, label='Follower Trajectory' if i == 0 else "")
    
    # 绘制领导者轨迹
    for i in range(len(trajectory_l[0])):
        x = [traj[i, 0] for traj in trajectory_l]
        y = [traj[i, 1] for traj in trajectory_l]
        z = [traj[i, 2] for traj in trajectory_l]
        ax.plot(x, y, z, color='red', alpha=0.3, label='Leader Trajectory' if i == 0 else "")
    
    # 绘制新加入的 agent 轨迹
    x = [traj[0] for traj in trajectory_n]
    y = [traj[1] for traj in trajectory_n]
    z = [traj[2] for traj in trajectory_n]
    ax.plot(x, y, z, color='green', alpha=0.3, label='New Agent Trajectory')
    
    # 绘制编队帧(一共三帧)
    first_frame = 1
    for frame, new_agent_frame in zip(formation_frames, new_agent_frames):
        # 绘制跟随者
        for i in range(len(frame[2:])):
            ax.scatter(frame[2:, 0][i], frame[2:, 1][i], frame[2:, 2][i], color='blue', s=20, alpha=0.6)
        
        # 绘制领导者
        for i in range(len(frame[:2])):
            ax.scatter(frame[:2, 0][i], frame[:2, 1][i], frame[:2, 2][i], color='red', s=20, alpha=0.6)

        # 绘制新加入的 agent
        ax.scatter(new_agent_frame[0], new_agent_frame[1], new_agent_frame[2], color='green', s=20, alpha=0.6)
        
        # 绘制连接线
        for j, k in edges[:-3]:
            ax.plot([frame[j-1, 0], frame[k-1, 0]], [frame[j-1, 1], frame[k-1, 1]], [frame[j-1, 2], frame[k-1, 2]], color='black', lw=1.0, alpha=1)

        if first_frame == 0:
            for j, k in edges[-3:]:
                ax.plot([new_agent_frame[0], frame[k-1, 0]], [new_agent_frame[1], frame[k-1, 1]], [new_agent_frame[2], frame[k-1, 2]], color='green', lw=1.0, alpha=1)


        first_frame = 0

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Formation Trajectory')
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
    plt.show()

plot_trajectories_with_formations()


# def plot_error_curves():
#     # 转换为numpy数组
#     leader_target = np.array(leader_target_positions)
#     leader_actual = np.array(leader_actual_positions)
#     follower_target = np.array(follower_target_positions)
#     follower_actual = np.array(follower_actual_positions)
    
#     # 计算误差
#     leader_error = leader_actual - leader_target[:, :leader_actual.shape[1], :]  # 确保维度匹配
#     follower_error = follower_actual - follower_target
    
#     # 时间轴
#     t_values = np.arange(0, leader_error.shape[0] * dt, dt)
    
#     # 创建图形和子图
#     plt.style.use('tableau-colorblind10')  # 使用专业配色风格
#     fig, axes = plt.subplots(3, 1, figsize=(10, 8))  # 调整画布大小
#     fig.subplots_adjust(top=0.85)  # 调整顶部空间
    
#     # 设置全局字体
#     plt.rc('font', family='serif', size=10)
    
#     # X方向误差
#     ax1 = axes[0]
#     for i in range(leader_error.shape[1]):
#         ax1.plot(t_values, leader_error[:, i, 0], '--', label=f'Leader {i+1}', linewidth=1.5)
#     for i in range(follower_error.shape[1]):
#         ax1.plot(t_values, follower_error[:, i, 0], label=f'Follower {i+1}', linewidth=1.5)
#     ax1.set_ylabel('X Error (m)')
#     ax1.grid(True, linestyle='--', alpha=0.7)
    
#     # Y方向误差
#     ax2 = axes[1]
#     for i in range(leader_error.shape[1]):
#         ax2.plot(t_values, leader_error[:, i, 1], '--', label=f'Leader {i+1}', linewidth=1.5)
#     for i in range(follower_error.shape[1]):
#         ax2.plot(t_values, follower_error[:, i, 1], label=f'Follower {i+1}', linewidth=1.5)
#     ax2.set_ylabel('Y Error (m)')
#     ax2.grid(True, linestyle='--', alpha=0.7)
    
#     # Z方向误差
#     ax3 = axes[2]
#     for i in range(leader_error.shape[1]):
#         ax3.plot(t_values, leader_error[:, i, 2], '--', label=f'Leader {i+1}', linewidth=1.5)
#     for i in range(follower_error.shape[1]):
#         ax3.plot(t_values, follower_error[:, i, 2], label=f'Follower {i+1}', linewidth=1.5)
#     ax3.set_ylabel('Z Error (m)')
#     ax3.set_xlabel('Time (s)')
#     ax3.grid(True, linestyle='--', alpha=0.7)
    
#     # 获取所有图例标签
#     handles, labels = ax1.get_legend_handles_labels()
    
#     # 计算图例列数，确保图例在一行显示
#     ncol = len(labels)
    
#     # 添加全局图例，放在图的上侧
#     fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=ncol, fontsize='medium', frameon=False)
    
#     # 调整布局，确保图例不被裁剪
#     plt.tight_layout(rect=[0, 0, 1, 0.88])  # 调整顶部空间
#     plt.show()

# plot_error_curves()