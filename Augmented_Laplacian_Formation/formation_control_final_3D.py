import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from self_functions import *



# 三维编队--正四面体
# D = np.array([
#     [ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0],
#     [-1,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  0],
#     [ 0,  0,  0,  0, -1,  0,  0,  1,  1,  1,  0,  0],
#     [ 0, -1,  0,  0,  0,  0,  0, -1,  0,  0,  1,  1],
#     [ 0,  0, -1,  0,  0, -1,  0,  0, -1,  0, -1,  0],
#     [ 0,  0,  0, -1,  0,  0, -1,  0,  0, -1,  0, -1]
# ])

# # 二维编队
# # D = np.array([
# #     [ 1,  1,  1,  0,  0,  0,  0],
# #     [-1,  0,  0,  1,  1,  0,  0],
# #     [ 0, -1,  0, -1,  0,  1,  1],
# #     [ 0,  0, -1,  0,  0, -1,  0],
# #     [ 0,  0,  0,  0, -1,  0, -1]
# # ])

# 维度 d: dimension is 3, agents moves in 3-dimensional world
d = 3

# 获取矩阵 D 的大小: n is num of agents, m is num of edges
# n, m = D.shape

# # 三维编队--正四面体
# r = np.array([
#     [0.5, 0.5, 0],
#     [0.5, -0.5, 0],
#     [-0.5, -0.5, 0],
#     [-0.5, 0.5, 0],
#     [0, 0, 1],
#     [0, 0, -1]
# ])

# # 二维编队
# # r = np.array([
# #     [0.5, 0.5, 0],
# #     [0.5, -0.5, 0],
# #     [0, 0, 0],
# #     [-1, 1, 0],
# #     [-1, -1, 0]
# # ])

# 通过关联矩阵 D 求取边集 edges
# edges = get_edges_from_incidence_matrix(D)

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

r = np.array([
    [0.05, 0, 1],
    [-0.05, 0, -1],
    [1, math.sqrt(3), 0.05],
    [1, -math.sqrt(3), -0.05],
    [-2, 0, 0]
])

n = r.shape[0]
m = edges.shape[0]

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
# Define via points
via = np.array([
    [0, 0, 0],          # 编队中心点位置
    [5, 0, 2.5],
    [10, 0, 5],
    [10, -10, 0],
    [10, -20, -5],
    [5, -20, 0],
    [0, -20, 5],
    [0, -10, 2.5],
    [0, 0, 0]
])

# Define Rotation
rot = [
    [l, 0],      # 旋转轴，旋转角度
    [l, 0],
    [l, -math.pi/2],
    [l, -math.pi/2],
    [l, -math.pi],
    [l, -math.pi],
    [l, -math.pi],
    [l, -3*math.pi/2],
    [l, -2*math.pi]
]

# rot = [
#     [[0, 0, 1], 0],      # 旋转轴，旋转角度
#     [[0, 0, 1], 0],
#     [[0, 0, 1], 0],
#     [[0, 0, 1], 0],
#     [[0, 0, 1], 0],
#     [[0, 0, 1], 0],
#     [[0, 0, 1], 0],
#     [[0, 0, 1], 0],
#     [[0, 0, 1], 0]
# ]


# Define Scale
sca = np.array([
    1,          # x、y、z 等比例缩放
    1,
    0.5,
    0.5,
    0.75,
    1.25,
    1,
    1,
    1,
])

# Initialize variables
ra = np.zeros((n, 3, via.shape[0])) # (agents num, dimension, via points num)
qvia = np.zeros((via.shape[0], 13))  # (via points num, translation(3) + rotation(9) + scale(3))

for j in range(via.shape[0]):
    translation = via[j, :]
    rotation = R.from_rotvec(rot[j][1] * np.array(rot[j][0])).as_matrix()
    scale = sca[j]
    qvia[j, :] = np.concatenate([translation, rotation.flatten(), [scale]])
    

# Generate trajectory
qr,dqr,ddqr,tr = mstraj_(qvia, 0.03, 0.2)



# 初始化
p_0 = r             # 初始位置，向量表示 (7, 3)
pF_0 = r[:n-2, :]   # 跟随者初始位置，向量表示 (5, 3)
pL_0 = r[n-2:]      # 领导者初始位置，向量表示 (2, 3)
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

# 添加轨迹记录和编队帧记录
trajectory_f = []  # 跟随者轨迹
trajectory_l = []  # 领导者轨迹
formation_frames = []  # 编队帧记录

# 用于记录目标位置和实际位置
leader_target_positions = []
leader_actual_positions = []
follower_target_positions = []
follower_actual_positions = []


# 创建图形和三维轴
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-5, 15)
ax.set_ylim(-25, 5)
ax.set_zlim(-5, 5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=20, azim=30)  # 设置初始视角

# 创建初始的散点图和线条
points_f, = ax.plot([], [], [], 'o', markersize=2, color='blue')  # 初始为空，将在init函数中设置
points_l, = ax.plot([], [], [], 'o', markersize=2, color='red')  # 初始为空，将在init函数中设置
lines = []
title_text = ax.text2D(0.5, 0.95, "", transform=ax.transAxes, fontsize=14, ha='center')
for j, k in edges:
    line, = ax.plot([], [], [], color='gray', lw=0.8, alpha=0.6)
    lines.append(line)

# 动画初始化函数
def init():
    # 设置初始位置
    points_f.set_data(p_0[:-2,0], p_0[:-2,1])
    points_f.set_3d_properties(p_0[:-2,2])
    points_l.set_data(p_0[-2:,0], p_0[-2:,1])
    points_l.set_3d_properties(p_0[-2:,2])
    for m, (j, k) in enumerate(edges):
        lines[m].set_data([p_0[j-1, 0], p_0[k-1, 0]], [p_0[j-1, 1], p_0[k-1, 1]])
        lines[m].set_3d_properties([p_0[j-1, 2], p_0[k-1, 2]])
    # title_text = ax.text2D(0.5, 0.95, "t = 0.0", transform=ax.transAxes, fontsize=14, ha='center')
    return (points_f, points_l) + tuple(lines) +  (title_text,)

# 动画更新函数
def update(frame):
    global p_t, pF_t, pL_t, v_t, vF_t, vL_t, loop, t, dt, aL, aF, qr, edges
    
    # 帧数超过控制点数时退出
    if loop >= qr.shape[0]:
        loop = qr.shape[0] - 1
    
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
    # pF_target_1 = (scale * rotation @ pF_0.T + translation.reshape(3, 1)).T   # (5, 3)
    pF_target = (-np.linalg.inv(Wff_total) @ Wfl_total @ (pL_t.flatten().reshape(-1, 1))).reshape(n-2, 3)
    # print("\n实际 Target：\n", pF_target_1)
    # print("通过 Leaders 推出的 Target：\n", pF_target_2)
    
    # 跟随者位置的目标速度
    vF_t_total = (-aF * (pF_t.flatten() + np.linalg.inv(Wff_total) @ Wfl_total @ pL_t.flatten())) - np.linalg.inv(Wff_total) @ Wfl_total @ vL_t_total
    vF_t = vF_t_total.reshape(n-2, 3)
    
    # 所有 agent 的位置和速度
    v_t = np.concatenate((vF_t, vL_t), axis=0)
    p_t += v_t * dt
    pF_t = p_t[:n-2, :]
    pL_t = p_t[n-2:, :]

    # 记录目标位置和实际位置
    leader_target_positions.append(pL_target.T.copy())
    follower_target_positions.append(pF_target.copy())
    leader_actual_positions.append(pL_t.copy())
    follower_actual_positions.append(pF_t.copy())

    # 更新散点位置
    points_f.set_data(p_t[:-2,0], p_t[:-2,1])
    points_f.set_3d_properties(p_t[:-2,2])
    points_l.set_data(p_t[-2:,0], p_t[-2:,1])
    points_l.set_3d_properties(p_t[-2:,2])
    # 更新线条位置
    for m, (j, k) in enumerate(edges):
        lines[m].set_data([p_t[j-1, 0], p_t[k-1, 0]], [p_t[j-1, 1], p_t[k-1, 1]])
        lines[m].set_3d_properties([p_t[j-1, 2], p_t[k-1, 2]])

    # 记录轨迹
    trajectory_f.append(p_t[:-2, :].copy())
    trajectory_l.append(p_t[-2:, :].copy())
    
    # 记录编队帧（每隔一定帧数记录一次）
    if loop % 500 == 0:  # 每隔50帧记录一次
        formation_frames.append(p_t.copy())

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
# ani.save("3D_formation_final_rotate_x.gif", writer="pillow", fps=30)
plt.show()


# 在全局变量中添加障碍物参数
obstacles = [
    {
        'center': np.array([10, -7.5, 1.25]),  # 障碍物中心位置
        'size': np.array([5, 0.1, 5]),     # 障碍物尺寸（长、宽、高）- 竖立的片状
        'hole_center': np.array([10, -7.5, 1.25]),  # 镂空区中心位置
        'hole_size': np.array([2, 0.1, 2])      # 镂空区尺寸（长、宽、高）
    }
]


def plot_trajectories_with_formations():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制障碍物
    for obs in obstacles:
        # 提取障碍物参数
        center = obs['center']
        size = obs['size']
        hole_center = obs['hole_center']
        hole_size = obs['hole_size']
        
        # 计算障碍物边界
        x_min, x_max = center[0] - size[0]/2, center[0] + size[0]/2
        z_min, z_max = center[2] - size[2]/2, center[2] + size[2]/2
        y = center[1]  # 竖立障碍物的Y坐标
        
        # 计算镂空区边界
        hx_min, hx_max = hole_center[0] - hole_size[0]/2, hole_center[0] + hole_size[0]/2
        hz_min, hz_max = hole_center[2] - hole_size[2]/2, hole_center[2] + hole_size[2]/2
        
        # 创建分区域网格 -------------------------------------------------
        # 左边区域（从障碍物左边界到镂空区左边界）
        if x_min < hx_min:
            x_left = np.linspace(x_min, hx_min, 2)
            z_left = np.linspace(z_min, z_max, 2)
            X_left, Z_left = np.meshgrid(x_left, z_left)
            Y_left = np.full_like(X_left, y)
            ax.plot_surface(X_left, Y_left, Z_left, color='gray', alpha=1)
        
        # 右边区域（从镂空区右边界到障碍物右边界）
        if hx_max < x_max:
            x_right = np.linspace(hx_max, x_max, 2)
            z_right = np.linspace(z_min, z_max, 2)
            X_right, Z_right = np.meshgrid(x_right, z_right)
            Y_right = np.full_like(X_right, y)
            ax.plot_surface(X_right, Y_right, Z_right, color='gray', alpha=1)
        
        # 下边区域（镂空区水平方向之间的下方区域）
        if z_min < hz_min:
            x_bottom = np.linspace(hx_min, hx_max, 2)
            z_bottom = np.linspace(z_min, hz_min, 2)
            X_bottom, Z_bottom = np.meshgrid(x_bottom, z_bottom)
            Y_bottom = np.full_like(X_bottom, y)
            ax.plot_surface(X_bottom, Y_bottom, Z_bottom, color='gray', alpha=1)
        
        # 上边区域（镂空区水平方向之间的上方区域）
        if hz_max < z_max:
            x_top = np.linspace(hx_min, hx_max, 2)
            z_top = np.linspace(hz_max, z_max, 2)
            X_top, Z_top = np.meshgrid(x_top, z_top)
            Y_top = np.full_like(X_top, y)
            ax.plot_surface(X_top, Y_top, Z_top, color='gray', alpha=1)

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
    
    # 绘制编队帧
    for frame in formation_frames:
        # 绘制跟随者
        for i in range(len(frame[:-2])):
            ax.scatter(frame[:-2, 0][i], frame[:-2, 1][i], frame[:-2, 2][i], color='blue', s=20, alpha=0.6)
        
        # 绘制领导者
        for i in range(len(frame[-2:])):
            ax.scatter(frame[-2:, 0][i], frame[-2:, 1][i], frame[-2:, 2][i], color='red', s=20, alpha=0.6)
        
        # 绘制连接线
        for j, k in edges:
            ax.plot([frame[j-1, 0], frame[k-1, 0]], [frame[j-1, 1], frame[k-1, 1]], [frame[j-1, 2], frame[k-1, 2]], color='black', lw=1.0, alpha=1)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Formation Trajectory')
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
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))  # 调整画布大小
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
    
    # Z方向误差
    ax3 = axes[2]
    for i in range(leader_error.shape[1]):
        ax3.plot(t_values, leader_error[:, i, 2], '--', label=f'Leader {i+1}', linewidth=1.5)
    for i in range(follower_error.shape[1]):
        ax3.plot(t_values, follower_error[:, i, 2], label=f'Follower {i+1}', linewidth=1.5)
    ax3.set_ylabel('Z Error (m)')
    ax3.set_xlabel('Time (s)')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
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