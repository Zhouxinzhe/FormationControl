import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# 初始化七个智能体的初始位置（正六边形加中心点）
r = 1
angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
pos_initial = np.array([[np.cos(angle)*r, np.sin(angle)*r, 0] for angle in angles])
pos_initial = np.vstack([pos_initial, [0, 0, 0]])  # 第七个点在中心

# 生成所有需要连接的点对
pairs = [(j, k) for j in range(7) for k in range(j+1, 7)]

# 创建图形和三维轴
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-2, 3)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=20, azim=30)  # 设置初始视角

# 创建初始的散点图和线条
points, = ax.plot([], [], [], 'o', markersize=8, color='blue')  # 初始为空，将在init函数中设置
lines = []
for j, k in pairs:
    line, = ax.plot([], [], [], color='gray', lw=0.8, alpha=0.6)
    lines.append(line)

# 动画初始化函数
def init():
    # 设置初始位置
    points.set_data(pos_initial[:,0], pos_initial[:,1])
    points.set_3d_properties(pos_initial[:,2])
    for m, (j, k) in enumerate(pairs):
        lines[m].set_data([pos_initial[j][0], pos_initial[k][0]], [pos_initial[j][1], pos_initial[k][1]])
        lines[m].set_3d_properties([pos_initial[j][2], pos_initial[k][2]])
    return (points,) + tuple(lines)

# 动画更新函数
def update(t):
    theta = 0.1 * t  # 旋转速度
    dx = 0.05 * t    # 平移速度
    pos = np.zeros((7, 3))
    for i in range(7):
        x_initial, y_initial, z_initial = pos_initial[i]
        # 绕Y轴旋转
        x_rot = x_initial * np.cos(theta) + z_initial * np.sin(theta)
        z_rot = -x_initial * np.sin(theta) + z_initial * np.cos(theta)
        y_rot = y_initial
        # 沿X轴平移
        x_new = x_rot + dx
        y_new = y_rot
        z_new = z_rot
        pos[i] = [x_new, y_new, z_new]
    
    # 更新散点位置
    points.set_data(pos[:,0], pos[:,1])
    points.set_3d_properties(pos[:,2])
    # 更新线条位置
    for m, (j, k) in enumerate(pairs):
        lines[m].set_data([pos[j][0], pos[k][0]], [pos[j][1], pos[k][1]])
        lines[m].set_3d_properties([pos[j][2], pos[k][2]])
    
    return (points,) + tuple(lines)

# 创建动画
ani = animation.FuncAnimation(
    fig, update,
    frames=np.linspace(0, 20, 200),  # 20秒动画，200帧
    init_func=init,
    blit=True,
    interval=50
)

plt.show()