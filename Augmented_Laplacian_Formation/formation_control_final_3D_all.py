# Description: 3D formation control with reconfiguration
# In this file, leader index is in the front, follower index is in the back

import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from self_functions import *



# 维度 d: dimension is 3, agents moves in 3-dimensional world
d = 3

# 三维编队--正三面体
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

actual_r = np.array([
    [0.5-1.5, 0.5+1, 0.05+1],
    [0.5-1.5, -1+1, -0.05+1],
    [-1-1.5, 0.5+1, 0+1],
    [0.05-1.5, 0.1+1, 0.5+1],
    [-0.05-1.5, -0.1+1, -0.5+1]
])

n = r.shape[0]
m = edges.shape[0]

plot_3d_formation(r, edges)


# 旋转轴
l_1 = np.array([0, 0, 1])
l_2 = np.array([0, 1, 0])  

# 生成权重矩阵 W, Wf, Wfl, Wff
W, Wf, Wfl, Wff = generate_weight_matrix_final(r, edges, l_1)
W_total = W.transpose(0, 2, 1, 3).reshape(3*n, 3*n)
Wf_total = W_total[3*2:, :]
Wfl_total = W_total[3*2:, :3*2]
Wff_total = W_total[3*2:, 3*2:]


"""
    Trajectory Generation
"""
# trajectory——设置三段轨迹，分别对应绕 z 轴旋转、绕 y 轴旋转、new agent join
traj_1 = [
    # via points    rotation        scale
    ([0, 0, 0],     [l_1, 0],             1),
    ([4, 0, 0],     [l_1, math.pi/4],     1),
    ([6, 2, 0],     [l_1, 0],             0.5),
    ([8, 2, 0],     [l_1, -math.pi/4],    0.5),
    ([10, 0, 0],    [l_1, 0],             1),
    ([12, 0, 0],    [l_1, 0],             1)
]
qvia_1 = np.zeros((len(traj_1), 13))  # (via points num, translation(3) + rotation(9) + scale(3))
for j in range(len(traj_1)):
    translation = traj_1[j][0]
    rotation = R.from_rotvec(traj_1[j][1][1] * np.array(traj_1[j][1][0])).as_matrix()
    scale = traj_1[j][2]
    qvia_1[j, :] = np.concatenate([translation, rotation.flatten(), [scale]])
qr_1,dqr_1,ddqr_1,tr_1 = mstraj_(qvia_1, 0.03, 0.2)

traj_2 = [
    # via points    rotation        scale
    ([12, 0, 0],    [l_2, 0],             1),
    ([14, 0, 0],    [l_2, -math.pi/4],    1),
    ([16, 0, 2],    [l_2, 0],             0.5),
    ([18, 0, 2],    [l_2, math.pi/4],     0.5),
    ([20, 0, 0],    [l_2, 0],             1),
    ([22, 0, 0],    [l_2, 0],             1)
]
qvia_2 = np.zeros((len(traj_2), 13))  # (via points num, translation(3) + rotation(9) + scale(3))
for j in range(len(traj_2)):
    translation = traj_2[j][0]
    rotation = R.from_rotvec(traj_2[j][1][1] * np.array(traj_2[j][1][0])).as_matrix()
    scale = traj_2[j][2]
    qvia_2[j, :] = np.concatenate([translation, rotation.flatten(), [scale]])
qr_2,dqr_2,ddqr_2,tr_2 = mstraj_(qvia_2, 0.03, 0.2)

traj_3 = [
    # via points    rotation        scale
    ([22, 0, 0],    [l_1, 0],          1),
    ([30, 0, 0],    [l_1, 0],          1),
]
qvia_3 = np.zeros((len(traj_3), 13))  # (via points num, translation(3) + rotation(9) + scale(3))
for j in range(len(traj_3)):
    translation = traj_3[j][0]
    rotation = R.from_rotvec(traj_3[j][1][1] * np.array(traj_3[j][1][0])).as_matrix()
    scale = traj_3[j][2]
    qvia_3[j, :] = np.concatenate([translation, rotation.flatten(), [scale]])
qr_3,dqr_3,ddqr_3,tr_3 = mstraj_(qvia_3, 0.03, 0.2)



"""
    Formation Initialization
"""
# 初始化
p_0 = actual_r    # 初始位置，向量表示 (7, 3)
pF_0 = p_0[2:, :]   # 跟随者初始位置，向量表示 (5, 3)
pL_0 = p_0[:2, :]      # 领导者初始位置，向量表示 (2, 3)
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
state = 1   # 1: 第一段轨迹，2: 第二段轨迹，3: 第三段轨迹
last_state = 1
t = 0
dt = 0.5
loop = 0
aL = 1  # 领导者控制参数
aF = 1  # 跟随者控制参数
threshold = 0.05     # 位置阈值, 用于判断是否到达目标位置
vL_threshold = 0.15   # leader 速度阈值, 用于判断是否到达目标速度
vF_threshold = 0.2   # follower 速度阈值, 用于判断是否到达目标速度

# 要加入的 agent 的参数设定
L = 1                 # 加入编队后与相邻两个 agent 的间距
thres_add = 0.1         # 加入编队的阈值，与目标位置距离小于该值时，认为到达指定位置，加入编队
getT = 0                # 是否到达目标位置的标志
last_getT = 0           # 上一次到达目标位置的标志
aN = 1                  # 要加入的 agent 的控制参数
us = np.array([0.0,0.0,0.0])        # 要加入的 agent 的速度
thres_us = 0.05          # 要加入的 agent 的速度阈值
ps_0 = np.array([30.0, 1, 2.8])  # 要加入的 agent 的初始位置
ps_t = ps_0.copy()      # t时刻，要加入的 agent 的位置
tarPos = ps_0.copy()         # 要加入的 agent 的目标位置


"""
    Record Variables
"""
# 添加轨迹记录和编队帧记录
trajectory_f = [pF_0.copy()]  # 跟随者轨迹
trajectory_l = [pL_0.copy()]  # 领导者轨迹
trajectory_n = [ps_0.copy()]  # 新加入的 agent 的轨迹
formation_frames = [p_0.copy()]  # 编队帧记录
new_agent_frames = [ps_0.copy()]  # 新加入的 agent 的帧记录

# 用于记录目标位置和实际位置
leader_target_positions = [r[:2, :].copy()]
leader_actual_positions = [pL_0.copy()]
follower_target_positions = [r[2:, :].copy()]
follower_actual_positions = [pF_0.copy()]
new_agent_target_positions = [ps_0.copy()]
new_agent_actual_positions = [ps_0.copy()]

# 用于记录是否形成目标编队
is_target_formation = [False]
is_getT = [False]


"""
    Animation
"""
# 自定义 3D 箭头类
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)


# 在全局变量中添加障碍物参数
obstacles = [
    {
        'center': np.array([7, 1.8, 0]),  # 障碍物中心位置
        'size': np.array([0.1, 3, 3]),     # 障碍物尺寸（长、宽、高）- 竖立的片状
        'hole_center': np.array([7, 1.8, 0]),  # 镂空区中心位置
        'hole_size': np.array([0.1, 1, 1])      # 镂空区尺寸（长、宽、高）
    },
    {
        'center': np.array([17, -0.2, 2]),  # 障碍物中心位置
        'size': np.array([0.1, 3, 3]),     # 障碍物尺寸（长、宽、高）- 竖立的片状
        'hole_center': np.array([17, -0.2, 2]),  # 镂空区中心位置
        'hole_size': np.array([0.1, 1, 1])      # 镂空区尺寸（长、宽、高）
    }
]

# 创建图形和三维轴
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-3, 34)
ax.set_ylim(-1, 3)
ax.set_zlim(-1, 3)
ax.set_xlabel('X', labelpad=20)
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_yticks([-1, 1, 3])
ax.set_zticks([-1, 1, 3])
ax.grid(False)
ax.set_box_aspect([37,10,10])
ax.view_init(elev=15, azim=-90)  # 设置初始视角
# 设置白色背景
# for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
#     axis.pane.set_color('white')
#     axis.line.set_color('black')


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
        if i >= m:  # 初始时刻的边数为 m，但后续加入新的agent后，边数会增加
            break
        lines[i].set_data([p_0[j-1, 0], p_0[k-1, 0]], [p_0[j-1, 1], p_0[k-1, 1]])
        lines[i].set_3d_properties([p_0[j-1, 2], p_0[k-1, 2]])
    
    # 绘制障碍物
    for obs in obstacles:
        # 提取障碍物参数
        center = obs['center']
        size = obs['size']
        hole_center = obs['hole_center']
        hole_size = obs['hole_size']
        
        # 计算障碍物边界
        y_min, y_max = center[1] - size[1]/2, center[1] + size[1]/2
        z_min, z_max = center[2] - size[2]/2, center[2] + size[2]/2
        x = center[0]  # 竖立障碍物的X坐标
        
        # 计算镂空区边界
        hy_min, hy_max = hole_center[1] - hole_size[1]/2, hole_center[1] + hole_size[1]/2
        hz_min, hz_max = hole_center[2] - hole_size[2]/2, hole_center[2] + hole_size[2]/2
        
        # 创建分区域网格 -------------------------------------------------
        # 左边区域（从障碍物左边界到镂空区左边界）
        if y_min < hy_min:
            y_left = np.linspace(y_min, hy_min, 2)
            z_left = np.linspace(z_min, z_max, 2)
            Y_left, Z_left = np.meshgrid(y_left, z_left)
            X_left = np.full_like(Y_left, x)
            ax.plot_surface(X_left, Y_left, Z_left, color='gray', alpha=0.6)
        
        # 右边区域（从镂空区右边界到障碍物右边界）
        if hy_max < y_max:
            y_right = np.linspace(hy_max, y_max, 2)
            z_right = np.linspace(z_min, z_max, 2)
            Y_right, Z_right = np.meshgrid(y_right, z_right)
            X_right = np.full_like(Y_right, x)
            ax.plot_surface(X_right, Y_right, Z_right, color='gray', alpha=0.6)
        
        # 下边区域（镂空区水平方向之间的下方区域）
        if z_min < hz_min:
            y_bottom = np.linspace(hy_min, hy_max, 2)
            z_bottom = np.linspace(z_min, hz_min, 2)
            Y_bottom, Z_bottom = np.meshgrid(y_bottom, z_bottom)
            X_bottom = np.full_like(Y_bottom, x)
            ax.plot_surface(X_bottom, Y_bottom, Z_bottom, color='gray', alpha=0.6)
        
        # 上边区域（镂空区水平方向之间的上方区域）
        if hz_max < z_max:
            y_top = np.linspace(hy_min, hy_max, 2)
            z_top = np.linspace(hz_max, z_max, 2)
            Y_top, Z_top = np.meshgrid(y_top, z_top)
            X_top = np.full_like(Y_top, x)
            ax.plot_surface(X_top, Y_top, Z_top, color='gray', alpha=0.6)

    return (points_f, points_l) + tuple(lines) +  (title_text,)

# 动画更新函数
def update(frame):
    global p_t, pF_t, pL_t, v_t, vF_t, vL_t, loop, t, dt, aL, aF, edges
    global n, edges, getT, ps_t, aN, thres_us, thres_add, L, us, tarPos, last_getT
    global W, Wf, Wfl, Wff, W_total, Wf_total, Wfl_total, Wff_total
    global state, last_state
    
    """
        Trajectory Update
    """
    if loop < qr_1.shape[0]:                    # 第一段轨迹
        translation = qr_1[loop, :3]
        rotation = qr_1[loop, 3:12].reshape(3, 3)
        scale = qr_1[loop, 12]
    elif loop < qr_1.shape[0] + qr_2.shape[0]:   # 第二段轨迹
        state = 2
        if last_state == 1:
            W, Wf, Wfl, Wff = generate_weight_matrix_final(p_t, edges, l_2)
            W_total = W.transpose(0, 2, 1, 3).reshape(3*n, 3*n)
            Wf_total = W_total[3*2:, :]
            Wfl_total = W_total[3*2:, :3*2]
            Wff_total = W_total[3*2:, 3*2:]
        translation = qr_2[loop-qr_1.shape[0], :3]
        rotation = qr_2[loop-qr_1.shape[0], 3:12].reshape(3, 3)
        scale = qr_2[loop-qr_1.shape[0], 12]
    elif loop < qr_1.shape[0] + qr_2.shape[0] + qr_3.shape[0]:    # 第三段轨迹
        state = 3
        if last_state == 2:
            W, Wf, Wfl, Wff = generate_weight_matrix_final(p_t, edges, l_1)
            W_total = W.transpose(0, 2, 1, 3).reshape(3*n, 3*n)
            Wf_total = W_total[3*2:, :]
            Wfl_total = W_total[3*2:, :3*2]
            Wff_total = W_total[3*2:, 3*2:]
        translation = qr_3[loop-qr_1.shape[0]-qr_2.shape[0], :3]
        rotation = qr_3[loop-qr_1.shape[0]-qr_2.shape[0], 3:12].reshape(3, 3)
        scale = qr_3[loop-qr_1.shape[0]-qr_2.shape[0], 12]
    else:       # 最后一帧
        loop = qr_1.shape[0] + qr_2.shape[0] + qr_3.shape[0] - 1
        translation = qr_3[loop-qr_1.shape[0]-qr_2.shape[0], :3]
        rotation = qr_3[loop-qr_1.shape[0]-qr_2.shape[0], 3:12].reshape(3, 3)
        scale = qr_3[loop-qr_1.shape[0]-qr_2.shape[0], 12]
    last_state = state

    """
        New Agent Update
    """
    # 要加入的 agent 的速度更新
    if state == 3 and getT == 0:
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
            W, Wf, Wfl, Wff = generate_weight_matrix_final(p_t, edges, l_1)
            W_total = W.transpose(0, 2, 1, 3).reshape(3*n, 3*n)
            Wf_total = W_total[3*2:, :]
            Wfl_total = W_total[3*2:, :3*2]
            Wff_total = W_total[3*2:, 3*2:]
            # 新增 lines
            for i in range(3):
                line, = ax.plot([], [], [], color='green', lw=0.8, alpha=0.6)
                lines.append(line)

    """
        Control Calculation 
    """
    # 领导者位置的目标位置
    pL_target = scale * rotation @ r[:2, :].T + translation.reshape(3, 1)   # (3, 2)
    
    # 领导者速度的目标速度
    for i in range(2):
        vL_t[i] = [aL * np.tanh(pL_target[0, i] - pL_t[i, 0]), aL * np.tanh(pL_target[1, i] - pL_t[i, 1]), aL * np.tanh(pL_target[2, i] - pL_t[i, 2])]
        if np.linalg.norm(vL_t[i]) > vL_threshold:
            vL_t[i] = vL_t[i] / np.linalg.norm(vL_t[i]) * vL_threshold
    vL_t_total = vL_t.flatten()

    # 跟随者位置的目标位置
    if getT == 0:
        pF_target = (scale * rotation @ r[2:, :].T + translation.reshape(3, 1)).T
        # pF_target = (-np.linalg.inv(Wff_total) @ Wfl_total @ (pL_t.flatten().reshape(-1, 1))).reshape(n-2, 3)
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
    if np.linalg.norm(vF_t[0]) > vF_threshold:
        vF_t[0] = vF_t[0] / np.linalg.norm(vF_t[0]) * vF_threshold
    if np.linalg.norm(vF_t[1]) > vF_threshold:
        vF_t[1] = vF_t[1] / np.linalg.norm(vF_t[1]) * vF_threshold
    if np.linalg.norm(vF_t[2]) > vF_threshold:
        vF_t[2] = vF_t[2] / np.linalg.norm(vF_t[2]) * vF_threshold
    if np.linalg.norm(us) > vF_threshold:
        us = us / np.linalg.norm(us) * vF_threshold

    """
        Update Position and Velocity
    """
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

    """
        Record
    """
    # 记录目标位置和实际位置
    leader_target_positions.append(pL_target.T.copy())
    follower_target_positions.append(pF_target.copy())
    leader_actual_positions.append(pL_t.copy())
    follower_actual_positions.append(pF_t.copy())
    new_agent_target_positions.append(tarPos.copy())
    new_agent_actual_positions.append(ps_t.copy())

    # 记录是否形成目标编队
    distance_L1 = np.linalg.norm(pL_target[:, 0] - pL_t[0, :])
    distance_L2 = np.linalg.norm(pL_target[:, 1] - pL_t[1, :])
    distance_F1 = np.linalg.norm(pF_target[0, :] - pF_t[0, :])
    distance_F2 = np.linalg.norm(pF_target[1, :] - pF_t[1, :])
    distance_F3 = np.linalg.norm(pF_target[2, :] - pF_t[2, :])
    form_target_formation = distance_L1 < threshold and distance_L2 < threshold and distance_F1 < threshold and distance_F2 < threshold and distance_F3 < threshold
    
    # 记录轨迹
    trajectory_f.append(pF_t.copy())
    trajectory_l.append(pL_t.copy())
    trajectory_n.append(ps_t.copy())
    
    # 记录编队帧（每隔一定帧数记录一次）
    if loop%200 == 100 or (getT == 1 and last_getT == 0) or loop == qr_1.shape[0] + qr_2.shape[0] + qr_3.shape[0]-2:
        if getT == 0:
            formation_frames.append(p_t.copy())
            new_agent_frames.append(ps_t.copy())
        else:
            formation_frames.append(p_t[:-1,:].copy())
            new_agent_frames.append(ps_t.copy())
        is_target_formation.append(form_target_formation)
        is_getT.append(getT)

    """
        Update Animation
    """
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
        if form_target_formation:
            lines[m].set_color('#7FFF00')
        else:
            lines[m].set_color('gray')

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
    frames=int(qr_1.shape[0] + qr_2.shape[0] + qr_3.shape[0]),
    init_func=init,
    blit=False,
    interval=1
)
# ani.save("3D_formation_final_reconfiguration.gif", writer="pillow", fps=30)
plt.show()



def plot_trajectories_with_formations():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-3, 34)
    ax.set_ylim(-1, 3)
    ax.set_zlim(-1, 3)
    ax.set_xlabel('X', labelpad=20)
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_yticks([-1, 1, 3])
    ax.set_zticks([-1, 1, 3])
    ax.grid(False)
    ax.set_box_aspect([37,10,10])
    ax.view_init(elev=15, azim=-90)  # 设置初始视角

    # 绘制障碍物
    for obs in obstacles:
        # 提取障碍物参数
        center = obs['center']
        size = obs['size']
        hole_center = obs['hole_center']
        hole_size = obs['hole_size']
        
        # 计算障碍物边界
        y_min, y_max = center[1] - size[1]/2, center[1] + size[1]/2
        z_min, z_max = center[2] - size[2]/2, center[2] + size[2]/2
        x = center[0]  # 竖立障碍物的X坐标
        
        # 计算镂空区边界
        hy_min, hy_max = hole_center[1] - hole_size[1]/2, hole_center[1] + hole_size[1]/2
        hz_min, hz_max = hole_center[2] - hole_size[2]/2, hole_center[2] + hole_size[2]/2
        
        # 创建分区域网格 -------------------------------------------------
        # 左边区域（从障碍物左边界到镂空区左边界）
        if y_min < hy_min:
            y_left = np.linspace(y_min, hy_min, 2)
            z_left = np.linspace(z_min, z_max, 2)
            Y_left, Z_left = np.meshgrid(y_left, z_left)
            X_left = np.full_like(Y_left, x)
            ax.plot_surface(X_left, Y_left, Z_left, color='gray', alpha=0.6)
        
        # 右边区域（从镂空区右边界到障碍物右边界）
        if hy_max < y_max:
            y_right = np.linspace(hy_max, y_max, 2)
            z_right = np.linspace(z_min, z_max, 2)
            Y_right, Z_right = np.meshgrid(y_right, z_right)
            X_right = np.full_like(Y_right, x)
            ax.plot_surface(X_right, Y_right, Z_right, color='gray', alpha=0.6)
        
        # 下边区域（镂空区水平方向之间的下方区域）
        if z_min < hz_min:
            y_bottom = np.linspace(hy_min, hy_max, 2)
            z_bottom = np.linspace(z_min, hz_min, 2)
            Y_bottom, Z_bottom = np.meshgrid(y_bottom, z_bottom)
            X_bottom = np.full_like(Y_bottom, x)
            ax.plot_surface(X_bottom, Y_bottom, Z_bottom, color='gray', alpha=0.6)
        
        # 上边区域（镂空区水平方向之间的上方区域）
        if hz_max < z_max:
            y_top = np.linspace(hy_min, hy_max, 2)
            z_top = np.linspace(hz_max, z_max, 2)
            Y_top, Z_top = np.meshgrid(y_top, z_top)
            X_top = np.full_like(Y_top, x)
            ax.plot_surface(X_top, Y_top, Z_top, color='gray', alpha=0.6)
    
    # 绘制跟随者轨迹
    for i in range(len(trajectory_f[0])):
        x = [traj[i, 0] for traj in trajectory_f]
        y = [traj[i, 1] for traj in trajectory_f]
        z = [traj[i, 2] for traj in trajectory_f]
        ax.plot(x, y, z, color='blue', alpha=0.3, label='Follower Trajectory' if i == 0 else "", lw=0.5)
    
    # 绘制领导者轨迹
    for i in range(len(trajectory_l[0])):
        x = [traj[i, 0] for traj in trajectory_l]
        y = [traj[i, 1] for traj in trajectory_l]
        z = [traj[i, 2] for traj in trajectory_l]
        ax.plot(x, y, z, color='red', alpha=0.3, label='Leader Trajectory' if i == 0 else "", lw=0.5)
    
    # 绘制新加入的 agent 轨迹
    x = [traj[0] for traj in trajectory_n]
    y = [traj[1] for traj in trajectory_n]
    z = [traj[2] for traj in trajectory_n]
    ax.plot(x, y, z, color='green', alpha=0.3, label='New Agent Trajectory', lw=0.5)
    
    # 绘制编队帧
    num = 0
    for frame, new_agent_frame, tar, getT in zip(formation_frames, new_agent_frames, is_target_formation, is_getT):
        if num == 7:    # 跳过第7帧，图好看一点
            num += 1
            continue
        
        # 绘制跟随者
        for i in range(len(frame[2:])):
            ax.scatter(frame[2:, 0][i], frame[2:, 1][i], frame[2:, 2][i], color='blue', s=10, alpha=0.6)
        
        # 绘制领导者
        for i in range(len(frame[:2])):
            ax.scatter(frame[:2, 0][i], frame[:2, 1][i], frame[:2, 2][i], color='red', s=10, alpha=0.6)

        # 绘制新加入的 agent
        ax.scatter(new_agent_frame[0], new_agent_frame[1], new_agent_frame[2], color='green', s=10, alpha=0.6)
        
        # 绘制连接线
        for j, k in edges[:-3]:
            if not tar:
                ax.plot([frame[j-1, 0], frame[k-1, 0]], [frame[j-1, 1], frame[k-1, 1]], [frame[j-1, 2], frame[k-1, 2]], color='gray', lw=1.0, alpha=0.6)
            else:
                ax.plot([frame[j-1, 0], frame[k-1, 0]], [frame[j-1, 1], frame[k-1, 1]], [frame[j-1, 2], frame[k-1, 2]], color='#7FFF00', lw=1.0, alpha=1)
            
        # 绘制新加入的 agent 与最后三个 agent 的连接线
        if getT:
            for j, k in edges[-3:]:
                ax.plot([new_agent_frame[0], frame[k-1, 0]], [new_agent_frame[1], frame[k-1, 1]], [new_agent_frame[2], frame[k-1, 2]], color='green', lw=1.0, alpha=1)

        # 绘制箭头
        frame_center = np.mean(frame, axis=0)
        if num == 1 or num == 2 or num == 5 or num == 6 or num == 8:
            arrow = Arrow3D(
                [frame_center[0], frame_center[0]],  # x 坐标
                [frame_center[1], frame_center[1]],  # y 坐标
                [frame_center[2], frame_center[2]+1],  # z 坐标
                mutation_scale=5,            # 箭头大小
                arrowstyle="-|>",            # 箭头样式
                color="red",               # 箭头颜色
                lw=2                         # 线宽
            )
            ax.add_artist(arrow)
        elif num == 3 or num == 4:
            arrow = Arrow3D(
                [frame_center[0], frame_center[0]],  # x 坐标
                [frame_center[1], frame_center[1]+1.5],  # y 坐标
                [frame_center[2], frame_center[2]],  # z 坐标
                mutation_scale=5,            # 箭头大小
                arrowstyle="-|>",            # 箭头样式
                color="red",               # 箭头颜色
                lw=2                         # 线宽
            )
            ax.add_artist(arrow)

        num += 1

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Formation Trajectory')
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
    plt.show()

plot_trajectories_with_formations()


def plot_error_curves():
    # 转换为numpy数组
    leader_target = np.array(leader_target_positions)
    leader_actual = np.array(leader_actual_positions)
    follower_target = np.array(follower_target_positions)
    follower_actual = np.array(follower_actual_positions)
    new_agent_target = np.array(new_agent_target_positions)
    new_agent_actual = np.array(new_agent_actual_positions)
    
    # 计算误差
    leader_error = leader_actual - leader_target[:, :leader_actual.shape[1], :]  # 确保维度匹配
    follower_error = follower_actual - follower_target
    new_agent_error = new_agent_actual - new_agent_target
    
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
    ax1.plot(t_values[920:], new_agent_error[920:, 0], label='New Agent', linewidth=1.5)
    ax1.set_ylabel('X Error (m)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Y方向误差
    ax2 = axes[1]
    for i in range(leader_error.shape[1]):
        ax2.plot(t_values, leader_error[:, i, 1], '--', label=f'Leader {i+1}', linewidth=1.5)
    for i in range(follower_error.shape[1]):
        ax2.plot(t_values, follower_error[:, i, 1], label=f'Follower {i+1}', linewidth=1.5)
    ax2.plot(t_values[920:], new_agent_error[920:, 1], label='New Agent', linewidth=1.5)
    ax2.set_ylabel('Y Error (m)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Z方向误差
    ax3 = axes[2]
    for i in range(leader_error.shape[1]):
        ax3.plot(t_values, leader_error[:, i, 2], '--', label=f'Leader {i+1}', linewidth=1.5)
    for i in range(follower_error.shape[1]):
        ax3.plot(t_values, follower_error[:, i, 2], label=f'Follower {i+1}', linewidth=1.5)
    ax3.plot(t_values[920:], new_agent_error[920:, 2], label='New Agent', linewidth=1.5)
    ax3.set_ylabel('Z Error (m)', labelpad=13)
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