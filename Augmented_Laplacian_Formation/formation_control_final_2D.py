import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from self_functions import *


# 二维编队
D = np.array([
    [ 1,  1,  1,  0,  0,  0,  0],
    [-1,  0,  0,  1,  1,  0,  0],
    [ 0, -1,  0, -1,  0,  1,  1],
    [ 0,  0, -1,  0,  0, -1,  0],
    [ 0,  0,  0,  0, -1,  0, -1]
])

# 维度 d: dimension is 3, agents moves in 3-dimensional world
d = 2

# 获取矩阵 D 的大小: n is num of agents, m is num of edges
n, m = D.shape

# 二维编队-nominal formation
r = np.array([
    [0.5, 0.5, 0],
    [0.5, -0.5, 0],
    [0, 0, 0],
    [-1, 1, 0],
    [-1, -1, 0]
])

# 二位编队-初始位置(与 nominal formation 有一定偏差)
actual_r = np.array([
    [-2, -1.0, 0],
    [-2, -1.8, 0],
    [-2.5, -1.5, 0],
    [-3, -1, 0],
    [-3, -2, 0]
])

# 通过关联矩阵 D 求取边集 edges
edges = get_edges_from_incidence_matrix(D)


plot_3d_formation(r, edges)


# 旋转轴
l = np.array([0, 0, 1])     

# 生成权重矩阵 W, Wf, Wfl, Wff
W, Wf, Wfl, Wff = generate_weight_matrix_final(r, edges, l)
W_total = W.transpose(0, 2, 1, 3).reshape(3*n, 3*n)
Wf_total = W_total[:3*(n-2), :]
Wfl_total = Wf_total[:3*(n-2), 3*(n-2):]
Wff_total = Wf_total[:3*(n-2), :3*(n-2)]



# trajectory
traj = [
    # via points    rotation        scale
    ([0, 0, 0],     [l, 0],             1),
    ([4, 0, 0],     [l, math.pi/4],     0.75),
    ([6, 2, 0],     [l, 0],             0.75),
    ([8, 2, 0],     [l, -math.pi/4],    0.75),
    ([10, 0, 0],    [l, 0],             0.75),
    ([12, 0, 0],    [l, 0],             0.5),
    ([18, 0, 0],    [l, 0],             0.5),
    ([23, 0, 0],    [l, 0],             1),
    # ([10, 0, 0],    [l, math.pi/2],     1),
    # ([10, 3, 0],    [l, math.pi/2],     0.35),
    # ([10, 6.5, 0],  [l, math.pi/2],     0.35),
    # ([10, 10, 0],   [l, math.pi],       1),
    # ([5, 10, 0],    [l, math.pi],       1),
    # ([0, 10, 0],    [l, 3*math.pi/2],   1),
    # ([0, 5, 0],     [l, 3*math.pi/2],   1),
    # ([0, 0, 0],     [l, 2*math.pi],     1)
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
qr,dqr,ddqr,tr = mstraj_(qvia, 0.01, 0.2)



# 初始化
p_0 = actual_r             # 初始位置，向量表示 (7, 3)
pF_0 = p_0[:n-2, :]   # 跟随者初始位置，向量表示 (5, 3)
pL_0 = p_0[n-2:]      # 领导者初始位置，向量表示 (2, 3)
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
aF = 2  # 跟随者控制参数
vL_threshold = 0.06  # 领导者速度阈值
vF_threshold = 0.1  # 跟随者速度阈值
threshold = 0.05     # 位置阈值, 用于判断是否到达目标位置

"""
    Record Variables
"""
# 添加轨迹记录和编队帧记录
trajectory_f = [pF_0.copy()]  # 跟随者轨迹
trajectory_l = [pL_0.copy()]  # 领导者轨迹
formation_frames = [p_0.copy()]  # 编队帧记录

# 用于记录目标位置和实际位置
leader_target_positions = [r[n-2:, :].copy()]
leader_actual_positions = [pL_0.copy()]
follower_target_positions = [r[:n-2, :].copy()]
follower_actual_positions = [pF_0.copy()]

# 用于记录是否形成目标编队
is_target_formation = [False]

"""
    Animation
"""
# 创建图形和三维轴
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.set_aspect('equal') # 设置x和y轴等比例
ax.set_xlim(-3.2, 25)
ax.set_ylim(-3.2, 3.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')

# 创建初始的散点图和线条
points_f, = ax.plot([], [], 'o', markersize=2, color='blue')  # 初始为空，将在init函数中设置
points_l, = ax.plot([], [], 'o', markersize=2, color='red')  # 初始为空，将在init函数中设置
lines = []
title_text = ax.text(0.5, 0.95, "", transform=ax.transAxes, fontsize=14, ha='center')
for j, k in edges:
    line, = ax.plot([], [], color='gray', lw=0.8, alpha=0.6)
    lines.append(line)

# 绘制障碍物
obstacles = [  # (左下角x坐标, 左下角y坐标, 宽度, 高度)
    (5.4, -0.5, 1.5, 0.8),  # 第一个障碍物
    (14, 0.7, 3.0, 1.0),   # 第二个障碍物
    (14, -1.7, 3.0, 1.0),    # 第三个障碍物
    (10, 1.3, 2, 1)     # 第四个障碍物
]
for (x, y, width, height) in obstacles:
    rect = Rectangle((x, y), width, height, 
                    facecolor='gray', edgecolor='black',
                    alpha=0.7, hatch='//')
    ax.add_patch(rect)


# 动画初始化函数
def init():
    # 设置初始位置
    points_f.set_data(pF_0[:,0], pF_0[:,1])
    points_l.set_data(pL_0[:,0], pL_0[:,1])
    for m, (j, k) in enumerate(edges):
        lines[m].set_data([p_0[j-1, 0], p_0[k-1, 0]], [p_0[j-1, 1], p_0[k-1, 1]])
    return (points_f, points_l) + tuple(lines) +  (title_text,)

# 动画更新函数
def update(frame):
    global p_t, pF_t, pL_t, v_t, vF_t, vL_t, loop, t, dt, aL, aF, qr, edges
    
    # 帧数超过控制点数时退出
    if loop >= qr.shape[0]:
        loop = qr.shape[0] - 1
    

    """
        Control Calculation
    """
    # loop 时的目标控制
    translation = qr[loop, :3]
    rotation = qr[loop, 3:12].reshape(3, 3)
    scale = qr[loop, 12]

    # 领导者位置的目标位置
    pL_target = scale * rotation @ r[n-2:, :].T + translation.reshape(3, 1)   # (3, 2)
    
    # 领导者速度的目标速度
    for i in range(2):
        vL_t[i] = [aL * np.tanh(pL_target[0, i] - pL_t[i, 0]), aL * np.tanh(pL_target[1, i] - pL_t[i, 1]), aL * np.tanh(pL_target[2, i] - pL_t[i, 2])]
        if np.linalg.norm(vL_t[i]) > vL_threshold:
            vL_t[i] = vL_t[i] / np.linalg.norm(vL_t[i]) * vL_threshold
    vL_t_total = vL_t.flatten()

    # 跟随者位置的目标位置
    pF_target = (scale * rotation @ r[:n-2, :].T + translation.reshape(3, 1)).T   # (5, 3)
    # pF_target = (-np.linalg.inv(Wff_total) @ Wfl_total @ (pL_t.flatten().reshape(-1, 1))).reshape(n-2, 3)
    
    # 跟随者位置的目标速度
    vF_t_total = (-aF * (pF_t.flatten() + np.linalg.inv(Wff_total) @ Wfl_total @ pL_t.flatten())) - np.linalg.inv(Wff_total) @ Wfl_total @ vL_t_total
    vF_t = vF_t_total.reshape(n-2, 3)
    if np.linalg.norm(vF_t[0]) > vF_threshold:
        vF_t[0] = vF_t[0] / np.linalg.norm(vF_t[0]) * vF_threshold
    if np.linalg.norm(vF_t[1]) > vF_threshold:
        vF_t[1] = vF_t[1] / np.linalg.norm(vF_t[1]) * vF_threshold
    if np.linalg.norm(vF_t[2]) > vF_threshold:
        vF_t[2] = vF_t[2] / np.linalg.norm(vF_t[2]) * vF_threshold
    
    """
        Update Position and Velocity
    """
    # 更新所有 agent 的位置和速度
    v_t = np.concatenate((vF_t, vL_t), axis=0)
    p_t += v_t * dt
    pF_t = p_t[:n-2, :]
    pL_t = p_t[n-2:, :]

    """
        Record
    """
    # 记录轨迹
    trajectory_f.append(p_t[:-2, :].copy())
    trajectory_l.append(p_t[-2:, :].copy())
    
    # 记录目标位置和实际位置
    leader_target_positions.append(pL_target.T.copy())
    follower_target_positions.append(pF_target.copy())
    leader_actual_positions.append(pL_t.copy())
    follower_actual_positions.append(pF_t.copy())

    # 记录是否形成目标编队
    distance_L1 = np.linalg.norm(pL_target[:, 0] - pL_t[0, :])
    distance_L2 = np.linalg.norm(pL_target[:, 1] - pL_t[1, :])
    distance_F1 = np.linalg.norm(pF_target[0, :] - pF_t[0, :])
    distance_F2 = np.linalg.norm(pF_target[1, :] - pF_t[1, :])
    distance_F3 = np.linalg.norm(pF_target[2, :] - pF_t[2, :])
    form_target_formation = distance_L1 < threshold and distance_L2 < threshold and distance_F1 < threshold and distance_F2 < threshold and distance_F3 < threshold

    # 记录编队帧（每隔一定帧数记录一次）
    if loop % 400 == 200 or loop == qr.shape[0] - 2:  # 每隔50帧记录一次
        formation_frames.append(p_t.copy())
        is_target_formation.append(form_target_formation)

    """
        Update Animation
    """
    # 更新散点位置
    points_f.set_data(p_t[:-2,0], p_t[:-2,1])
    points_l.set_data(p_t[-2:,0], p_t[-2:,1])
    
    # 更新线条位置
    for m, (j, k) in enumerate(edges):
        lines[m].set_data([p_t[j-1, 0], p_t[k-1, 0]], [p_t[j-1, 1], p_t[k-1, 1]])
        if form_target_formation:
            lines[m].set_color('#7FFF00')
        else:
            lines[m].set_color('gray')

    # 更新时间
    t += dt
    loop += 1
    
    # 更新标题
    title_text.set_text(f"t = {t:.1f}")
    
    return (points_f, points_l) + tuple(lines) + (title_text,)

# 创建动画
ani = animation.FuncAnimation(
    fig, update,
    frames=int(qr.shape[0]),
    init_func=init,
    blit=True,
    interval=1
)
# ani.save("2D_formation_final_2.gif", writer="pillow", fps=30)
plt.show()









def plot_trajectories_with_formations():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal') # 设置x和y轴等比例
    
    # 绘制障碍物
    for (x, y, width, height) in obstacles:
        rect = Rectangle((x, y), width, height, 
                        facecolor='gray', edgecolor='black',
                        alpha=0.7, hatch='//')
        ax.add_patch(rect)

    # 绘制跟随者轨迹
    for i in range(len(trajectory_f[0])):
        x = [traj[i, 0] for traj in trajectory_f]
        y = [traj[i, 1] for traj in trajectory_f]
        ax.plot(x, y, color='blue', alpha=0.5, label='Follower Trajectory' if i == 0 else "", lw = 0.5)
    
    # 绘制领导者轨迹
    for i in range(len(trajectory_l[0])):
        x = [traj[i, 0] for traj in trajectory_l]
        y = [traj[i, 1] for traj in trajectory_l]
        ax.plot(x, y, color='red', alpha=0.5, label='Leader Trajectory' if i == 0 else "", lw = 0.5)
    
    # 绘制编队帧
    for frame, tar in zip(formation_frames, is_target_formation):
        # 绘制连接线
        for j, k in edges:
            if tar: ax.plot([frame[j-1, 0], frame[k-1, 0]], [frame[j-1, 1], frame[k-1, 1]], color='#7FFF00', lw=2, alpha=1, zorder=1)
            else: ax.plot([frame[j-1, 0], frame[k-1, 0]], [frame[j-1, 1], frame[k-1, 1]], color='gray', lw=2, alpha=1, zorder=1)

        # 绘制跟随者
        for i in range(len(frame[:-2])):
            ax.scatter(frame[:-2, 0][i], frame[:-2, 1][i], color='blue', s=30, alpha=1, zorder=2)
        
        # 绘制领导者
        for i in range(len(frame[-2:])):
            ax.scatter(frame[-2:, 0][i], frame[-2:, 1][i], color='red', s=30, alpha=1, zorder=2)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    # ax.set_title('Formation Trajectory')
    ax.legend(loc='upper right')
    plt.show()

plot_trajectories_with_formations()


def plot_error_curves():
    # 转换为numpy数组
    leader_target = np.array(leader_target_positions)
    leader_actual = np.array(leader_actual_positions)
    follower_target = np.array(follower_target_positions)
    follower_actual = np.array(follower_actual_positions)
    
    # 计算误差
    leader_error = leader_actual - leader_target[:, :leader_actual.shape[1], :]  # 确保维度匹配
    follower_error = follower_actual - follower_target
    
    # 时间轴
    t_values = np.arange(0, leader_error.shape[0] * dt, dt)
    
    # 创建图形和子图
    plt.style.use('tableau-colorblind10')  # 使用专业配色风格
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))  # 调整画布大小
    fig.subplots_adjust(top=0.85)  # 调整顶部空间
    
    # 设置全局字体
    plt.rc('font', family='serif', size=10)
    
    # X方向误差
    ax1 = axes[0]
    for i in range(leader_error.shape[1]):
        ax1.plot(t_values, leader_error[:, i, 0], '--', label=f'Leader {i+1}', linewidth=1.5)
    for i in range(follower_error.shape[1]):
        ax1.plot(t_values, follower_error[:, i, 0], label=f'Follower {i+1}', linewidth=1.5)
    ax1.set_ylabel('X Error (m)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Y方向误差
    ax2 = axes[1]
    for i in range(leader_error.shape[1]):
        ax2.plot(t_values, leader_error[:, i, 1], '--', label=f'Leader {i+1}', linewidth=1.5)
    for i in range(follower_error.shape[1]):
        ax2.plot(t_values, follower_error[:, i, 1], label=f'Follower {i+1}', linewidth=1.5)
    ax2.set_ylabel('Y Error (m)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlabel('Time (s)')
    
    # 获取所有图例标签
    handles, labels = ax1.get_legend_handles_labels()
    
    # 计算图例列数，确保图例在一行显示
    ncol = len(labels)
    
    # 添加全局图例，放在图的上侧
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=ncol, fontsize='medium', frameon=False)
    
    # 调整布局，确保图例不被裁剪
    plt.tight_layout(rect=[0, 0, 1, 0.88])  # 调整顶部空间
    plt.show()

plot_error_curves()