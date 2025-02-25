import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import sympy as sp

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