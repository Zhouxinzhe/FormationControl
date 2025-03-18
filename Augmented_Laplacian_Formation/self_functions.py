import numpy as np
from scipy.special import comb
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sympy import symbols, Eq, solve
from itertools import product


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
    ax.scatter(r[:-2, 0], r[:-2, 1], r[:-2, 2], c='blue', marker='o', s=100, label='Followers')
    ax.scatter(r[-2:, 0], r[-2:, 1], r[-2:, 2], c='red', marker='o', s=100, label='Leaders')
    
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



def solve_sphere_intersection(x0, y0, z0, x1, y1, z1, x2, y2, z2, r):
    """
    输入三个球中心坐标和半径r，求解三个球交点。
    返回值为可能的两个交点（若只有一个交点或不存在则分别处理）。
    """
    # 定义三个球中心的向量形式
    P1 = np.array([x0, y0, z0], dtype=float)
    P2 = np.array([x1, y1, z1], dtype=float)
    P3 = np.array([x2, y2, z2], dtype=float)
    
    # 构造局部坐标系，首先计算 x 轴单位向量 ex = (P2-P1)/||P2-P1||
    ex = P2 - P1
    d = np.linalg.norm(ex)
    if d == 0:
        raise ValueError("P1和P2不能重合！")
    ex = ex / d
    
    # 计算 i = dot(ex, P3-P1)
    P3_P1 = P3 - P1
    i = np.dot(ex, P3_P1)
    
    # 计算中间变量，用于构造 y 轴单位向量
    temp = P3_P1 - i * ex
    temp_norm = np.linalg.norm(temp)
    if temp_norm == 0:
        raise ValueError("三个点共线，无法唯一确定交点！")
    ey = temp / temp_norm
    
    # 构造 z 轴单位向量，使用叉乘保证右手坐标系
    ez = np.cross(ex, ey)
    
    # 计算 j = dot(ey, P3-P1)
    j = np.dot(ey, P3_P1)
    
    # 利用前两个球的方程，得到在局部坐标系中交点在 x 轴上的投影
    x = d / 2.0
    
    # 利用第三个球的方程，求解 y 的坐标
    # 公式： (x-i)^2+(y-j)^2+z^2 = r^2  且 x^2+y^2+z^2=r^2, 得
    # (x-i)^2+(y-j)^2 = r^2 - z^2 = r^2 - (r^2 - x^2-y^2) = x^2+y^2
    # 展开化简可得： y = (i^2 + j^2 - d*i) / (2*j)
    if np.abs(j) < 1e-6:
        raise ValueError("j接近0，可能导致除0错误，说明第三个点在x轴方向上，与前两个点共线。")
    y = (i**2 + j**2 - d*i) / (2*j)
    
    # 求 z 值，根据 x^2+y^2+z^2 = r^2
    squared_term = r**2 - x**2 - y**2
    if squared_term < 0:
        raise ValueError("无实数解，三个球没有公共交点！")
    z_val = np.sqrt(squared_term)
    
    # 两个可能的解，分别对应 z = +sqrt(...) 和 z = -sqrt(...)
    sol1_local = np.array([x, y,  z_val])
    sol2_local = np.array([x, y, -z_val])
    
    # 转换回全局坐标
    # P = P1 + ex * x + ey * y + ez * z
    sol1_global = P1 + sol1_local[0] * ex + sol1_local[1] * ey + sol1_local[2] * ez
    sol2_global = P1 + sol2_local[0] * ex + sol2_local[1] * ey + sol2_local[2] * ez
    
    # 若 z_val 非零则有两个解，否则只有一个解
    if np.isclose(z_val, 0):
        return [sol1_global]
    else:
        return [sol1_global, sol2_global]


nbr_0 = -1
nbr_1 = -1
nbr_2 = -1

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
    nbr0 : int
        1-based index of the first neighbor
    nbr1 : int
        1-based index of the second neighbor
    nbr2 : int
        1-based index of the third neighbor
    
    """
    n = p.shape[0]  # 编队中节点的数量
    
    # 计算要加入的节点的当前位置与编队中各个节点的距离
    Dist = np.zeros(n)
    for i in range(n):
        Dist[i] = np.linalg.norm(ps - p[i])
    sorted_indices = np.argsort(Dist)
    min_three_indices = sorted_indices[:3]
    nbr0 = min_three_indices[0] # 0-based index
    nbr1 = min_three_indices[1]
    nbr2 = min_three_indices[2]

    # 求解要加入的节点的目标位置
    x_0, y_0, z_0 = p[nbr0]
    x_1, y_1, z_1 = p[nbr1]
    x_2, y_2, z_2 = p[nbr2]

    # change nbr to 1-based index
    nbr0 += 1
    nbr1 += 1
    nbr2 += 1

    solutions = solve_sphere_intersection(x_0, y_0, z_0, x_1, y_1, z_1, x_2, y_2, z_2, L)

    if len(solutions) == 0:
        return None, nbr0, nbr1, nbr2
    elif len(solutions) == 1:
        pt = solutions[0]
    else:
        sol_1 = solutions[0]
        sol_2 = solutions[1]

        # 选择距离要加入节点最近的解
        sum_dist_1 = np.linalg.norm(sol_1 - ps)
        sum_dist_2 = np.linalg.norm(sol_2 - ps)
        if sum_dist_1 > sum_dist_2:
            pt = sol_2
        else:
            pt = sol_1
    
    return pt, nbr0, nbr1, nbr2



def update_edge(edge, nbr1, nbr2, nbr3, n):
    """
    更新边集，当新的节点 node 加入编队后将 node 与 nbr1 和 node 与 nbr2 之间的边加入到边集中。
    新加入的节点加在最后。

    Parameters
    ----------
    edge : np.ndarray
        编队的边集，(m, 2)
    nbr1 : int
        1-based index of the first neighbor
    nbr2 : int
        1-based index of the second neighbor
    nbr3 : int
        1-based index of the third neighbor
    n : int
        new index of the new node
    
    Returns
    -------
    edge_new : np.ndarray
        更新后的边集，(m+3, 2)
    """
    edge_new = np.vstack((edge, np.array([n, nbr1]), np.array([n, nbr2]), np.array([n, nbr3])))
    
    return edge_new