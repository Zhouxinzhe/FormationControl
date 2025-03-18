import numpy as np
from scipy.special import comb
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import sympy as sp
from itertools import product

def plot_curve(y_values, title="Y值曲线图", x_label="X", y_label="Y"):
    """
    绘制给定y值列表的曲线图。

    Parameters
    ----------
        y_values : np.ndarray
            y值列表。
    """
    # 默认x值为从0开始的整数序列
    x_values = list(range(len(y_values)))

    # 绘制曲线
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')  # 添加点标记和线条样式
    plt.title(title)  # 添加标题
    plt.xlabel(x_label)  # 添加X轴标签
    plt.ylabel(y_label)  # 添加Y轴标签
    plt.grid(True)  # 添加网格
    plt.show()



def plot_multiple_curves(y_values_lists, labels=None, title="多条曲线图", x_label="X", y_label="Y"):
    """
    在同一个图上绘制多条曲线，并添加图例。

    Parameters
    ----------
        y_values_lists : np.ndarray 
            每个参数是一个y值列表，表示一条曲线。
        labels : list, optional 
            一个列表，包含每条曲线的标签（可选）。
        title : str, optional
            图表标题（可选）。
        x_label : str, optional
            X轴标签（可选）。
        y_label : str, optional
            Y轴标签（可选）。
    """
    # 如果没有提供labels，生成默认标签
    if labels is None:
        labels = [f"Curve {i + 1}" for i in range(len(y_values_lists))]

    # 确保labels数量与曲线数量匹配
    if len(labels) != len(y_values_lists):
        print("标签数量与曲线数量不匹配。")
        return

    # 绘制每条曲线
    for y_values, label in zip(y_values_lists, labels):
        x_values = list(range(len(y_values)))  # 自动生成x值
        plt.plot(x_values, y_values, label=label)  # 绘制曲线并添加标签

    # 添加图例
    plt.legend()

    # 添加标题和轴标签
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # 添加网格
    plt.grid(True)

    # 显示图表
    plt.show()



def draw_current_state(r, edge):
    """
    绘制当前编队的状态

    Parameters
    ----------
    r : np.ndarray
        编队中每个节点的坐标，(n, 2) 
    edge : np.ndarray
        编队的边集，每一条边表示为 (node1, node2).

    Returns
    -------
    None
    """
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

    # 绘制要加入的节点（红色点）
    # plt.plot(Vec_ps[0], Vec_ps[1], 'r.', markersize=30)

    # 设置坐标轴范围
    plt.axis([-5, 5, -5, 5])

    # 设置title
    plt.title("Current State")

    # 显示图形
    plt.show()



def SrchNbr(i, edge):
    """
    Searches for the neighbors of a node i in a graph. Node indices begin from 1.

    Parameters
    ----------
    i : int
        The index of the node of interest. 
    edge : list of tuples
        The edge list of the graph, where each edge is represented as a tuple (node1, node2).

    Returns
    -------
    Nbr : list
        A list of neighbors of node i.

    Notes
    -----
    This function assumes that the edge list is undirected. If the graph is directed,
    the function will still work but may not capture all relevant neighbors depending
    on the edge direction.

    Example
    -------
    >>> edge_list = [(1, 2), (1, 3), (2, 4)]
    >>> neighbors = SrchNbr(1, edge_list)
    >>> print(neighbors)
    [2, 3]

    See Also
    --------
    SrchNbrDirected: A variant for directed graphs (if needed).
    """
    Nbr = []
    for j in range(len(edge)):
        if edge[j][0] == i:
            Nbr.append(int(edge[j][1]))
        elif edge[j][1] == i:
            Nbr.append(int(edge[j][0]))
    return Nbr



def compute_weight_i(NBR, r_0, i, n):
    """
    Compute the weight vector Wi based on the given neighborhood matrix NBR,
    position matrix r_0, index i, and total number of agents n. Node indices begin from 1.

    Parameters
    ----------
    NBR : list
        Neighborhood list indicating neighbors of agent i.
    r_0 : np.ndarray
        Positions of the agents in complex number.
    i : int
        Index of the current agent.
    n : int
        Total number of agents.

    Returns
    -------
    Wi : numpy.ndarray
        Weight vector (1 x n) for agent i.
    """
    m = len(NBR)  # Number of neighbors of agent i
    Tnum = int(comb(m, 2))  # Number of combinations of m taken 2 at a time
    Yita = np.zeros((Tnum, n), dtype=np.complex128)  # Initialize Yita matrix

    k = 0  # Index for Yita
    for s in range(m - 1):
        for t in range(s + 1, m):
            Yita[k, NBR[s]-1] = r_0[NBR[t]-1] - r_0[i-1]
            Yita[k, NBR[t]-1] = r_0[i-1] - r_0[NBR[s]-1]
            k += 1

    # Compute Wi by summing over Yita
    Wi = np.sum(Yita, axis=0)
    return Wi



def compute_weight_i_matrix(NBR, r, i, n, d):
    """
    Compute the weight vector Wi based on the given neighborhood matrix NBR,
    position matrix r, index i, and total number of agents n. Node indices begin from 1.

    Parameters
    ----------
    NBR : list
        Neighborhood list indicating neighbors of agent i.
    r : np.ndarray
        Positions of the agents in vectors.
    i : int
        Index of the current agent.
    n : int
        Total number of agents.
    d : int
        Dimension

    Returns
    -------
    Wi : numpy.ndarray
        Weight vector (d x nd) for agent i.
    """
    m = len(NBR)  # Number of neighbors of agent i
    Tnum = int(comb(m, 2))  # Number of combinations of m taken 2 at a time
    Yita_x = np.zeros((Tnum, n*d))  # Initialize Yita matrix
    Yita_y = np.zeros((Tnum, n*d))
    Wi = np.zeros((d, n*d))

    k = 0  # Index for Yita
    for s in range(m - 1):
        for t in range(s + 1, m):
            Yita_x[k, d*(NBR[s]-1):d*(NBR[s]-1)+2] = r[NBR[t]-1, 0] - r[i-1, 0], -(r[NBR[t]-1, 1] - r[i-1, 1])
            Yita_y[k, d*(NBR[s]-1):d*(NBR[s]-1)+2] = r[NBR[t]-1, 1] - r[i-1, 1], r[NBR[t]-1, 0] - r[i-1, 0]
            Yita_x[k, d*(NBR[t]-1):d*(NBR[t]-1)+2] = r[i-1, 0] - r[NBR[s]-1, 0], -(r[i-1, 1] - r[NBR[s]-1, 1])
            Yita_y[k, d*(NBR[t]-1):d*(NBR[t]-1)+2] = r[i-1, 1] - r[NBR[s]-1, 1], r[i-1, 0] - r[NBR[s]-1, 0]
            k += 1

    # Compute Wi by summing over Yita
    Yita_x = np.sum(Yita_x, axis=0)
    Yita_y = np.sum(Yita_y, axis=0)
    Wi[0, :] = Yita_x
    Wi[1, :] = Yita_y 
    return Wi



def rot2(theta):
    """2D rotation matrix"""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])



def mstraj(viapoints, dist_step=0.1):
    """
    Generate a trajectory from initial point to multiple via points using linear interpolation.

    Parameters:
        viapoints (ndarray): A set of via points, one per row (m x n).
        dist_step (float, optional): distance step. Defaults to 0.1.

    Returns:
        q (ndarray): Trajectory positions (m x n).
    """
    q0 = viapoints[0, :]  # Use the first via point as initial position
    q = [q0]  # Initialize trajectory with the initial point

    for i in range(1, viapoints.shape[0]):
        start = viapoints[i - 1]
        end = viapoints[i]
        dist = end - start
        num_steps = int(np.ceil(np.linalg.norm(dist) / dist_step))  # Number of steps based on distance

        # Linear interpolation between start and end points
        segment = np.linspace(start, end, num_steps, endpoint=True)
        q.extend(segment[1:])  # Append interpolated points (excluding the first point)

    q = np.array(q)  # Convert list to numpy array
    return q



def mstraj_(qvia, dist_step, dt):
    """
    Generate trajectory from via points.

    Parameters:
        qvia (numpy.ndarray): Via points (each row is a via point).
        dist (float): Distance step.
        dt (float): Time step.

    Returns:
        q (numpy.ndarray): Trajectory positions.
        qd (numpy.ndarray): Trajectory velocities.
        qdd (numpy.ndarray): Trajectory accelerations.
        t (numpy.ndarray): Time sequence.
    """
    qlen = qvia.shape[1]  # Number of dimensions
    q = mstraj(viapoints=qvia,  dist_step=dist_step)
    qd = np.vstack((np.diff(q, axis=0) / dt, np.zeros((1, qlen))))
    qdd = np.vstack((np.diff(qd, axis=0) / dt, np.zeros((1, qlen))))
    t = np.arange(0, q.shape[0] * dt, dt)
    return q, qd, qdd, t



def find_circle_intersections(x0, y0, x1, y1, L):
    """
    计算两个圆的交点，几何方法。
    
    Parameters
    ----------
        (x0, y0): 第一个圆的圆心
        (x1, y1): 第二个圆的圆心
        L: 两个圆的半径
    
    Returns
    -------
        交点列表 [(x1, y1), (x2, y2)] 或 None
    """
    # 计算圆心之间的距离
    d = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    
    # 检查是否有交点
    if d > 2 * L or d == 0:
        return None
    
    # 计算 h
    h = np.sqrt(L**2 - (d / 2)**2)

    # 计算圆心中点坐标
    x_m = (x0 + x1) / 2
    y_m = (y0 + y1) / 2

    # 计算三角函数
    sin_theta = (y1 - y0) / d
    cos_theta = (x1 - x0) / d

    # 计算交点坐标
    x_1 = x_m + h * sin_theta
    y_1 = y_m - h * cos_theta
    x_2 = x_m - h * sin_theta
    y_2 = y_m + h * cos_theta

    return [(x_1, y_1), (x_2, y_2)]



def FindTargetPostion(p, ps, edge, L):
    """
    通过当前编队各个节点的位置、及其拓扑结构（边集），以及要加入的节点的当前位置、
    要加入节点的目标位置与当前编队的位置约束，求取要加入的节点的目标位置。
    
    Parameters
    ----------
    p : np.ndarray
        当前编队各个节点的位置复数表示，(n,)
    ps : np.ndarray
        要加入的节点的当前位置复数表示，(1,)
    edge : np.ndarray
        编队的边集，(m, 2)
    L : float
        要加入节点的目标位置与当前编队的位置约束
    
    Returns
    -------
    pt : np.ndarray
        要加入的节点的目标位置复数表示，(1,)
    min_idx: int

    min_NBR: int

    Notes
    -----
    1. min_idx 指的是与要加入的节点的当前位置距离最小的节点的索引。这里的索引最后转换为1-based index。
    2. min_NBR 指的是与要加入的节点的当前位置距离最小的节点的邻居节点中，与与要加入的节点的当前位置距离最小的节点的索引。
       这里的索引最后转换为1-based index。
    
    """
    n = p.shape[0]  # 编队中节点的数量
    
    # 计算要加入的节点的当前位置与编队中各个节点的距离
    Dist = np.zeros(n)
    for i in range(n):
        Dist[i] = np.abs(ps - p[i])
    min_dist = np.min(Dist)     # 距离最小值
    min_idx = np.argmin(Dist)   # 距离最小值对应的节点的索引
    min_idx += 1    # Convert to 1-based index

    # 求距离最小值对应的节点的邻居节点
    NBR = SrchNbr(min_idx, edge)
    m = len(NBR)    # 邻居节点的数量
    NBR_dist = np.zeros(m)  # 邻居节点与要加入的节点的距离
    for i in range(m):
        NBR_dist[i] = np.abs(ps - p[NBR[i] - 1])    # Convert to 0-based index
    min_NBR_dist = np.min(NBR_dist)     # 邻居节点与要加入的节点的距离最小值
    min_NBR_idx = np.argmin(NBR_dist)   # 邻居节点与要加入的节点的距离最小值对应的节点的索引
    min_NBR = NBR[min_NBR_idx]          # 邻居节点与要加入的节点的距离最小值对应的节点, which is 1-based index

    # 求解要加入的节点的目标位置
    # x, y = sp.symbols('x y')
    x_0 = p[min_idx - 1].real
    y_0 = p[min_idx - 1].imag
    x_1 = p[min_NBR - 1].real
    y_1 = p[min_NBR - 1].imag
    # equations = [
    #     (x - x_0)**2 + (y - y_0)**2 - L**2,
    #     (x - x_1)**2 + (y - y_1)**2 - L**2
    # ]
    # solutions = sp.solve(equations, (x, y))     # 两个圆相交
    solutions = find_circle_intersections(x_0, y_0, x_1, y_1, L)

    if len(solutions) == 0:
        return None, min_idx, min_NBR
    elif len(solutions) == 1:
        pt = solutions[0][0] + solutions[0][1] * 1j
    else:
        sol_1 = solutions[0][0] + solutions[0][1] * 1j
        sol_2 = solutions[1][0] + solutions[1][1] * 1j

        # 选择距离当前编队中所有节点最远的解
        sum_dist_1 = 0
        sum_dist_2 = 0
        for i in range(n):
            # print(sol_1, p[i], np.abs(sol_1 - p[i]))
            sum_dist_1 += np.abs(sol_1 - p[i])
            sum_dist_2 += np.abs(sol_2 - p[i])
        # print("sum_dist_1:", sum_dist_1)
        # print("sum_dist_2:", sum_dist_2)
        if sum_dist_1 > sum_dist_2:
            pt = sol_2
        else:
            pt = sol_1
    
    pt = np.complex128(pt)
    return pt, min_idx, min_NBR



def generate_complex_weight_matrix(edge, r_complex):
    """
    基于编队的边集和节点的位置，生成复权重矩阵。

    Parameters
    ----------
    edge : np.ndarray
        编队的边集，(m, 2)
    r_complex : np.ndarray
        节点的位置复数表示，(n,)
    
    Returns
    -------
    W : np.ndarray
        复权重矩阵，(n, n)
    Wf : np.ndarray
        复权重矩阵的前 n-2 行，(n-2, n)
    Wfl : np.ndarray
        复权重矩阵的前 n-2 行的后两列，(n-2, 2)
    Wff : np.ndarray
        复权重矩阵的前 n-2 行的前 n-2 列，(n-2, n-2)
    
    """
    n = r_complex.shape[0]  # 编队中节点的数量  
    W = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        NBR = SrchNbr(i+1, edge)    # Convert to 1-based index
        Wi = compute_weight_i(NBR, r_complex, i+1, n)
        W[i] = Wi
        sum_i = 0
        for j in range(n):
            sum_i -= Wi[j]
        W[i, i] = sum_i
    
    Wf = W[0:n-2, :]
    Wfl = W[0:n-2, n-2:]
    Wff = W[0:n-2, 0:n-2]

    return W, Wf, Wfl, Wff



def update_edge(edge, nbr1, nbr2):
    """
    更新边集，当新的节点 node 加入编队后将 node 与 nbr1 和 node 与 nbr2 之间的边加入到边集中。
    新的节点 node 的索引为 1，原本所有节点的索引依次加 1。

    Parameters
    ----------
    edge : np.ndarray
        编队的边集，(m, 2)
    nbr1 : int
        节点1的索引
    nbr2 : int
        节点2的索引
    
    Returns
    -------
    edge_new : np.ndarray
        更新后的边集，(m+1, 2)
    """
    for i in range(edge.shape[0]):
        edge[i, 0] += 1
        edge[i, 1] += 1
    edge_new = np.vstack((np.array([1, nbr1+1]), np.array([1, nbr2+1]), edge))  # 因为原本所有节点的索引都加了 1
    
    return edge_new



def get_edges_from_incidence_matrix(D):
    """
    从关联矩阵中获取边集。节点索引从 1 开始。

    Parameters
    ----------
    D : np.ndarray
        关联矩阵，(n, m)
    
    Returns
    -------
    edge : np.ndarray
        边集，(m, 2)
    """
    n, m = D.shape
    # 找到 D 中非零元素的索引
    non_zero_indices_row, non_zero_indices_col = np.where(D != 0)
    # 计算 non_zero_indices
    non_zero_indices = sorted(non_zero_indices_row + non_zero_indices_col * n + 1)
    # 将索引转换为 Mx2 的矩阵
    edges = np.mod(np.reshape(non_zero_indices, (m, 2)), n)
    # 将 0 替换为 n
    edges[edges == 0] = n
    return edges



def plot_3d_formation(r, edges):
    """
    绘制三维编队图，给定节点坐标和边连接信息。
    
    Parameters
    ----------
    r (np.ndarray): 节点坐标，形状为 (n, 3)，其中 n 是节点数。
    edges (np.ndarray): 边连接信息，形状为 (m, 2)，其中 m 是边数。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制节点
    ax.scatter(r[:, 0], r[:, 1], r[:, 2], c='blue', marker='o', s=100, label='Agents')
    
    # 绘制边
    line_segments = []
    for edge in edges:
        # 转换为0索引
        i, j = edge[0] - 1, edge[1] - 1
        line_segments.append([r[i], r[j]])
    
    line_collection = Line3DCollection(line_segments, color='gray', linestyle='-', linewidth=1)
    ax.add_collection(line_collection)
    
    # 设置坐标轴范围
    ax.set_xlim(np.min(r[:, 0]) - 1, np.max(r[:, 0]) + 1)
    ax.set_ylim(np.min(r[:, 1]) - 1, np.max(r[:, 1]) + 1)
    ax.set_zlim(np.min(r[:, 2]) - 1, np.max(r[:, 2]) + 1)
    
    # 添加图例
    ax.legend()
    
    plt.show()



def compute_transformation_matrix_3D(v):
    """
    计算将给定向量 v 旋转为 x 轴上单位向量的变换矩阵 A。
        1.将向量v旋转到xy平面上，旋转轴为与 [x,y,0]^T垂直的那根轴
        2.将旋转到xy平面上的向量再绕z轴旋转转到x轴正半轴 
        3.通过放缩，乘一个标量，使得最终的向量落到[1,0,0]^T。
    
    Parameters
    ----------
    v : np.ndarray
        3D向量 (x, y, z)
    
    Returns
    -------
    A : np.ndarray
        变换矩阵

    """
    x, y, z = v
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        raise ValueError("Zero vector cannot be transformed.")

    # Step 1: Rotate v to the xy-plane
    xy_norm = np.sqrt(x**2 + y**2)
    if xy_norm == 0:
        R1 = np.eye(3)  # If x=y=0, no need for the first rotation
    else:
        axis_1 = np.array([y, -x, 0]) / xy_norm  # Normalized rotation axis
        theta_1 = -np.arctan2(z, xy_norm)  # Rotation angle
        R1 = R.from_rotvec(theta_1 * axis_1).as_matrix()

    # Apply first rotation
    v_rot1 = R1 @ v

    # Step 2: Rotate in xy-plane to align with x-axis
    x2, y2, _ = v_rot1
    theta_2 = -np.arctan2(y2, x2)
    R2 = np.array([
        [np.cos(theta_2), -np.sin(theta_2), 0],
        [np.sin(theta_2),  np.cos(theta_2), 0],
        [0, 0, 1]
    ])

    # Step 3: Scale the vector to unit length
    scale = 1 / norm_v
    A = scale * R2 @ R1
    return A



def compute_weight_i_3D(NBR, r, i, n):
    """
    Compute the weight vector Wi based on the given neighborhood matrix NBR,
    position matrix r_0, index i, and total number of agents n. Node indices begin from 1.

    Parameters
    ----------
    NBR : list
        Neighborhood list indicating neighbors of agent i.
    r : np.ndarray        
        Positions of the agents 
        节点坐标，(n, 3)
    i : int
        Index of the current agent.
    n : int
        Total number of agents.

    Returns
    -------
    Wi : numpy.ndarray
        Weight vector for agent i.
        (n, 3, 3)
        Wij 是一个 3x3 的矩阵
    """
    m = len(NBR)  # Number of neighbors of agent i
    Tnum = int(comb(m, 2))  # Number of combinations of m taken 2 at a time
    Yita = np.zeros((Tnum, n, 3, 3))  # Initialize Yita matrix

    k = 0  # Index for Yita
    for s in range(m - 1):
        for t in range(s + 1, m):
            Yita[k, NBR[s]-1] = compute_transformation_matrix_3D(r[NBR[s]-1] - r[i-1])
            Yita[k, NBR[t]-1] = compute_transformation_matrix_3D(r[i-1] - r[NBR[t]-1])
            k += 1

    # Compute Wi by summing over Yita
    Wi = np.sum(Yita, axis=0)
    return Wi



def generate_weight_matrix_3D(r, edges):
    """
    生成三维编队的权重矩阵。

    Parameters
    ----------
    r : np.ndarray
        节点坐标，(n, 3)
    edges : np.ndarray
        边连接信息，(m, 2)
    
    Returns
    -------
    W : np.ndarray
        权重矩阵，(n, n, 3, 3)
    Wf : np.ndarray
        权重矩阵的前 n-2 行，(n-2, n, 3, 3)
    Wfl : np.ndarray
        权重矩阵的前 n-2 行的后两列，(n-2, 2, 3, 3)
    Wff : np.ndarray
        权重矩阵的前 n-2 行的前 n-2 列，(n-2, n-2, 3, 3)
    """
    n = r.shape[0]  # 编队中节点的数量  
    W = np.zeros((n, n, 3, 3))
    for i in range(n):
        NBR = SrchNbr(i+1, edges)    # Convert to 1-based index
        Wi = compute_weight_i_3D(NBR, r, i+1, n)    # Wi (n, 3, 3)
        W[i] = Wi
        sum_i = 0
        for j in range(n):
            sum_i -= Wi[j]
        W[i, i] = sum_i
    
    Wf = W[0:n-2, :]
    Wfl = W[0:n-2, n-2:]
    Wff = W[0:n-2, 0:n-2]

    return W, Wf, Wfl, Wff



def skew_symmetric_matrix(v):
    """
    计算向量 v 的反对称矩阵。

    Parameters
    ----------
    v : np.ndarray
        3D向量
    
    Returns
    -------
    np.ndarray
        3x3反对称矩阵
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])



def compute_weight_ij_ik(p0, p1, p2, l):
    """
    计算 agents p0, p1, p2 间的权重矩阵，使得 w_01 @ (p1 - p0) + w_02 @ (p2 - p0) = 0。

    Parameters
    ----------
    p0 : np.ndarray (3,)
        agent 0 的位置
    p1 : np.ndarray (3,)
        agent 1 的位置
    p2 : np.ndarray (3,)
        agent 2 的位置
    l : np.ndarray  (3,)
        旋转轴
    
    Returns
    -------
    w_01 : np.ndarray
        agent 0 和 agent 1 之间的权重矩阵
    w_02 : np.ndarray
        agent 0 和 agent 2 之间的权重矩阵
    
    """
    l = l.reshape(-1, 1)  # (3, 1)

    # 定义向量差
    v1 = p1 - p0
    v2 = p2 - p0

    # 定义斜对称矩阵函数
    def skew_symmetric_matrix(v):
        return np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])

    # 构造矩阵 w_01 和 w_02 的表达式
    def construct_w(a, b, c, l):
        return a * np.eye(3) + b * (l @ l.T) + c * skew_symmetric_matrix(l.flatten())

    # 定义目标方程
    def target_equation(a01, b01, c01, a02, b02, c02):
        w_01 = construct_w(a01, b01, c01, l)
        w_02 = construct_w(a02, b02, c02, l)
        return w_01 @ v1 + w_02 @ v2

    # 构造线性方程组
    # 我们需要解的是 target_equation(a01, b01, c01, a02, b02, c02) = 0
    # 这是一个由 3 个方程组成的系统（每个分量为 0）

    # 展开方程，得到系数矩阵和右侧向量
    # 这里我们手动展开方程，构造系数矩阵

    # 定义变量顺序：a01, b01, c01, a02, b02, c02
    num_vars = 6
    num_equations = 3

    # 初始化系数矩阵和右侧向量
    A = np.zeros((num_equations, num_vars))
    b = np.zeros(num_equations)

    # 计算每个变量的系数
    for i in range(num_equations):
        # 对于每个方程（对应结果向量的每个分量）
        # 计算 w_01 @ v1 + w_02 @ v2 的第 i 个分量
        
        # 展开 w_01 @ v1 的第 i 个分量
        # w_01 = a01 * I + b01 * (l @ l.T) + c01 * skew_symmetric_matrix(l)
        # 因此，w_01 @ v1 的第 i 个分量是：
        # a01 * (I @ v1)[i] + b01 * ((l @ l.T) @ v1)[i] + c01 * (skew_symmetric_matrix(l) @ v1)[i]
        
        # 同理，w_02 @ v2 的第 i 个分量是：
        # a02 * (I @ v2)[i] + b02 * ((l @ l.T) @ v2)[i] + c02 * (skew_symmetric_matrix(l) @ v2)[i]
        
        # 因此，整个方程的第 i 个分量为：
        # a01 * (I @ v1)[i] + b01 * ((l @ l.T) @ v1)[i] + c01 * (skew_symmetric_matrix(l) @ v1)[i] +
        # a02 * (I @ v2)[i] + b02 * ((l @ l.T) @ v2)[i] + c02 * (skew_symmetric_matrix(l) @ v2)[i] = 0
        
        # 计算各项系数
        A[i, 0] = (np.eye(3) @ v1)[i]  # a01 的系数
        A[i, 1] = ((l @ l.T) @ v1)[i]  # b01 的系数
        A[i, 2] = (skew_symmetric_matrix(l.flatten()) @ v1)[i]  # c01 的系数
        A[i, 3] = (np.eye(3) @ v2)[i]  # a02 的系数
        A[i, 4] = ((l @ l.T) @ v2)[i]  # b02 的系数
        A[i, 5] = (skew_symmetric_matrix(l.flatten()) @ v2)[i]  # c02 的系数

    # 使用奇异值分解求解齐次方程组的非零解
    U, s, Vh = np.linalg.svd(A)
    # 计算矩阵 A 的秩
    rank_A = np.sum(s > 1e-10)  # 考虑数值误差，设置一个阈值
    n = A.shape[1]
    # 零空间的维数
    nullity_A = n - rank_A
    # 获取零空间的基向量
    null_space_basis = Vh[-nullity_A:, :]

    # 定义系数的取值范围
    coefficient_range = np.linspace(-1, 1, 10)  # 可以根据需要调整范围和精度

    # 生成所有可能的系数组合
    coefficient_combinations = product(coefficient_range, repeat=nullity_A)

    # 筛选出能使 w_01 和 w_02 可逆的解
    valid_solution = None
    for coefficients in coefficient_combinations:
        # 计算基向量的线性组合
        solution = np.zeros(num_vars)
        for i, coef in enumerate(coefficients):
            solution += coef * null_space_basis[i]
        
        a01, b01, c01, a02, b02, c02 = solution
        w_01 = construct_w(a01, b01, c01, l)
        w_02 = construct_w(a02, b02, c02, l)
        det_w01 = np.linalg.det(w_01)
        det_w02 = np.linalg.det(w_02)
        if np.abs(det_w01) > 1e-10 and np.abs(det_w02) > 1e-10:
            valid_solution = solution
            break

    if valid_solution is None:
        # print("未找到能使 w_01 和 w_02 可逆的解。")
        a01, b01, c01, a02, b02, c02 = Vh[-1, :]
    else:
        # 提取解
        a01, b01, c01, a02, b02, c02 = valid_solution

        # 输出结果
        # print(f"a01 = {a01}")
        # print(f"b01 = {b01}")
        # print(f"c01 = {c01}")
        # print(f"a02 = {a02}")
        # print(f"b02 = {b02}")
        # print(f"c02 = {c02}")

    # 验证解是否满足方程
    w_01 = construct_w(a01, b01, c01, l)
    w_02 = construct_w(a02, b02, c02, l)
    # result = w_01 @ v1 + w_02 @ v2
    # print("\n验证结果（应接近零向量）:")
    # print(result)

    # 验证 w_01 和 w_02 是否可逆
    # print(f"w_01 的行列式: {np.linalg.det(w_01)}")
    # print(f"w_02 的行列式: {np.linalg.det(w_02)}")

    return w_01, w_02



def compute_weight_i_final(NBR, r, i, l):
    """
    Compute the weight vector Wi based on the given neighborhood matrix NBR,
    position matrix r_0, index i, and total number of agents n. Node indices begin from 1.

    Parameters
    ----------
    NBR : list
        Neighborhood list indicating neighbors of agent i.
    r : np.ndarray        
        Positions of the agents 
        节点坐标，(n, 3)
    i : int
        Index of the current agent.
    l : np.ndarray  (3,)
        旋转轴

    Returns
    -------
    Wi : numpy.ndarray
        Weight vector for agent i.
        (n, 3, 3)
        Wij 是一个 3x3 的矩阵
    """
    m = len(NBR)  # Number of neighbors of agent i
    n = r.shape[0]  # Number of  agents
    Tnum = int(comb(m, 2))  # Number of combinations of m taken 2 at a time
    Yita = np.zeros((Tnum, n, 3, 3))  # Initialize Yita matrix

    k = 0  # Index for Yita
    for s in range(m - 1):
        for t in range(s + 1, m):
            Yita[k, NBR[s]-1], Yita[k, NBR[t]-1] = compute_weight_ij_ik(r[i-1], r[NBR[s]-1], r[NBR[t]-1], l)
            k += 1

    # Compute Wi by summing over Yita
    Wi = np.sum(Yita, axis=0)
    return Wi

def generate_weight_matrix_final(r, edges, l):
    """
    生成三维编队的权重矩阵。

    Parameters
    ----------
    r : np.ndarray
        节点坐标，(n, 3)
    edges : np.ndarray
        边连接信息，(m, 2)
    l : np.ndarray  (3,)
        旋转轴
        
    Returns
    -------
    W : np.ndarray
        权重矩阵，(n, n, 3, 3)
    Wf : np.ndarray
        权重矩阵的前 n-2 行，(n-2, n, 3, 3)
    Wfl : np.ndarray
        权重矩阵的前 n-2 行的后两列，(n-2, 2, 3, 3)
    Wff : np.ndarray
        权重矩阵的前 n-2 行的前 n-2 列，(n-2, n-2, 3, 3)
    """
    n = r.shape[0]  # 编队中节点的数量  
    W = np.zeros((n, n, 3, 3))
    for i in range(n):
        NBR = SrchNbr(i+1, edges)    # Convert to 1-based index
        Wi = compute_weight_i_final(NBR, r, i+1, l)    # Wi (n, 3, 3)
        W[i] = Wi
        sum_i = 0
        for j in range(n):
            sum_i -= Wi[j]
        W[i, i] = sum_i
    
    Wf = W[0:n-2, :]
    Wfl = W[0:n-2, n-2:]
    Wff = W[0:n-2, 0:n-2]

    return W, Wf, Wfl, Wff