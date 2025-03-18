# Formation Control

## Preliminary

* 双向图的复拉普拉斯矩阵 $L$：
  $$
  L(i, j) = 
  \begin{cases} 
  -w_{ij} & \text{if } i \neq j \text{ and } j \in \mathcal{N}_i \\
  0 & \text{if } i \neq j \text{ and } j \notin \mathcal{N}_i \\
  \sum_{k \in \mathcal{N}_i} w_{ik} & \text{if } i = j
  \end{cases}
  $$
  实际代码中的权重矩阵 $W=-L$

* 节点坐标定义 $p$ : $\vec{p} = (𝑝_1, . . . , 𝑝_𝑛) \in C^n$，是复数表示



## 复拉普拉斯矩阵构建

* 每一个节点 𝑖 的复约束方程:
  $$
  \sum_{j \in \mathcal{N}_i} w_{ij}(p_j - p_i) = 0
  $$
  $\Rightarrow$
  $$
  L\vec{p} = 0
  $$

* **框架相似**：在二维空间内，对于同一个 $L$，如果
  $$
  ker(L) = \{c_11_n+c_2\vec{p}:c_1,c_2\in C\}
  $$
  则满足复约束方程 $L\vec{p}=0$ 的所有 $G(\vec{p})$ 是相似的

  （说人话，二维平面上，希望编队平移、旋转、放缩后，仍满足 $L\vec{p}=0$）

  （显然，上面构建出来的 $L$ 有两个特征向量 $1_n$和$\vec{p}$）

* **构建**：

  1. 节点 𝑖 有正好两个邻居节点
     $$
     [𝑤_{𝑖𝑗},\ 𝑤_{𝑖𝑘}] = 𝛼_𝑖[𝑝_𝑘 − 𝑝_𝑖,\ 𝑝_𝑖 − 𝑝_𝑗] 
     $$

  2. 节点 i 有的邻居节点多于两个，假设有 m 个邻居节点

     * 从中选取两个节点 $i_j,\ i_k$，构建一个 m 维向量 $\zeta_h$
       $$
       𝜁_ℎ = [0, 0, . . . , 𝑝_{𝑖_𝑘} − 𝑝_𝑖
       , . . . , 𝑝_𝑖 − 𝑝_{𝑖_𝑗}
       , 0, 0, . . . , 0] 
       $$

     * 重复构建 $C_m^2$ 个$\zeta_h$

     * 求和：
       $$
       [w_{ii_1}, w_{ii_2}, \ldots, w_{ii_m}] = \sum_{h=1}^{C_m^2} \alpha_i^h \zeta_h
       $$



## 编队控制

* **Followers 位置可以由 Leaders 确定**
  $$
  L = \begin{bmatrix} L_{ff} & L_{fl} \\ L_{lf} & L_{ll} \end{bmatrix} \quad r = \begin{bmatrix} r_F \\ r_L \end{bmatrix}
  $$
  $\Rightarrow$
  $$
  L_{ff}r_F + L_{fl}r_L = 0 \\
  r_F = -L_{ff}^{-1}L_{fl}r_L
  $$
  假定初始 $G(\vec{p})$ 构建出的 $L$ 中 $L_{ff}$ 是可逆的（即对初始 $G(\vec{p})$ 有一定的要求）

* **Leaders** 运动控制

  假设编队中只有 Leaders 知道自己的当前位置 $p_i$ 与目标位置 $p_i^*$，即只有 Leaders 做路径的规划

  这里直接使用一阶控制：
  $$
  v_i = -\tanh(x_i - x_i^*) - i \tanh(y_i - y_i^*) + \dot{p}_i^*
  $$

  $$
  \begin{cases}
  \dot{x}_i = -\tanh(x_i - x_i^*) + \dot{x}_i^* \\
  \dot{y}_i = -\tanh(y_i - y_i^*) + \dot{y}_i^*
  \end{cases}
  $$

  代码中，将目标位置的目标速度 $\dot{p}_i^*$ 设置为 0

* **Followers** 运动控制

  仅需要与邻居节点的相对位置信息，以及速度信息就可以设计控制律使得整个编队移动过程是稳定
  $$
  \dot{p}_i = -\frac{1}{\gamma_i} \sum_{j \in \mathcal{N}_i} w_{ij} [\alpha_1 (p_i - p_j) - \dot{p}_j], i \in \mathcal{V}_f
  $$



# 论文整体思路

* 编队入侵 formation invasion
  * 基于 Complex Laplace Matrix 的 Formation Control，给出具备节点加入不影响原本节点的控制策略和特性的证明
    * 可以得到，如果已知编队的拓扑与控制律，能够加入新的节点（节点*）
  * 拓扑推断 / 系统辨识
  * 控制律推断
    * 通过推断得到编队拓扑和控制律的预测，新节点入侵后与节点*间的控制误差





# abstract

多智能体系统的编队控制一直是非常火热的研究领域。但当前的大多数研究关注于对于既定的多智能体系统的编队控制，鲜有人研究如何实现编队入侵。本文提出了一种对基于复拉普拉斯矩阵控制的编队进行编队入侵的方法。首先，我们将复拉普拉斯矩阵控制编队建立为一阶线性时不变系统。 然后，基于编队系统的轨迹信息，使用 ordinary least-squares (OLS) estimator 对系统参数进行预测，并基于此进行控制律的推断。同时对于系统参数预测结果的收敛性进行了理论分析 。最后进行了仿真实验，将编队入侵的控制效果与原始的编队控制进行了对比，以验证上述方法的有效性。

kimi 润色：

多智能体系统的编队控制一直是研究热点。然而，当前大多数研究集中在既定多智能体系统的编队控制上，而对编队入侵的研究较少。本文提出了一种基于复拉普拉斯矩阵控制的编队入侵方法。首先，我们将复拉普拉斯矩阵控制的编队建模为一阶线性时不变系统。然后，基于编队系统的轨迹信息，利用普通最小二乘估计器（OLS）对系统参数进行预测，并基于此推导控制律。此外，本文对系统参数预测结果的收敛性进行了理论分析。最后，通过仿真实验验证了所提方法的有效性，对比了编队入侵控制与原始编队控制的效果。

kimi 翻译：

Formation control of multi-agent systems has been a popular research topic. However, most existing studies focus on the formation control of established multi-agent systems, while research on formation intrusion is relatively limited. This paper proposes a method for formation intrusion based on complex Laplacian matrix control. First, we model the complex Laplacian matrix-controlled formation as a first-order linear time-invariant (LTI) system. Then, based on the trajectory information of the formation system, we use the ordinary least-squares (OLS) estimator to predict the system parameters and derive the control law. Additionally, we conduct a theoretical analysis of the convergence of the system parameter prediction results. Finally, simulation experiments are conducted to verify the effectiveness of the proposed method by comparing the control performance of formation intrusion with that of the original formation control.



# Introduction



In recent years, the formation control of multi-agent systems has garnered significant attention due to its wide range of applications in various fields, such as drone swarms, robotic cooperation, and intelligent transportation systems（文献）. The ability to maintain a desired formation structure is crucial for the successful execution of coordinated tasks. Consequently, numerous studies have been devoted to the development of formation control strategies for multi-agent systems under different scenarios and constraints.（文献：有哪些控制策略。。。） These efforts have led to substantial progress in the design and implementation of control algorithms that ensure the stability and robustness of predefined formations.

* Y. Li and J. Tong, "3D multi-UAV coupled formation control based on backstepping control method," 2024 IEEE 6th Advanced Information Management, Communicates, Electronic and Automation Control Conference (IMCEC), Chongqing, China, 2024, pp. 1517-1521, doi: 10.1109/IMCEC59810.2024.10575758. keywords: {Couplings;Backstepping;Three-dimensional displays;Trajectory tracking;Autonomous aerial vehicles;Formation control;Stability analysis;UAV;UAV formation control;backstepping control;coupling control;trajectory tracking},

  * **改进反步控制法**：引入积分项，提升**轨迹跟踪**性能

  * 分布式控制

    | 控制对象               | 主要任务               | 控制方法                         | 主要方程                                                     |
    | ---------------------- | ---------------------- | -------------------------------- | ------------------------------------------------------------ |
    | **Leader（领航者）**   | 轨迹跟踪               | 反步控制（Backstepping Control） | Uz=z¨d+c1e1+k1e1intU_z = \ddot{z}_d + c_1 e_1 + k_1 e_1^{\text{int}}Uz=z¨d+c1e1+k1e1int |
    | **Follower（跟随者）** | 跟随 Leader + 维持编队 | 耦合控制（Coupling Control）     | ui=−∑aijKieiju_i = -\sum a_{ij} K_i e_{ij}ui=−∑aijKieij      |

* B. Liu, S. Yang, Q. Li and Y. Tu, "Safe Formation Control For Micro-UAV Using Control Barrier Functions," 2024 China Automation Congress (CAC), Qingdao, China, 2024, pp. 3933-3938, doi: 10.1109/CAC63892.2024.10864752. keywords: {Automation;Autonomous aerial vehicles;Formation control;Safety;Maintenance;Collision avoidance;Micro-UAV formation;super-twisting sliding mode control;safe formation control},

  * Tracking：Uses an **adaptive super-twisting control algorithm** to ensure finite-time convergence.
  * Safe Control：Incorporates a **CBF-based safe formation strategy** to prevent collisions.

However, despite the extensive research on formation control, the problem of formation intrusion has received relatively less attention. Formation intrusion refers to the scenario where an external agent or a group of agents attempts to join an existing formation dynamically. （编队入侵，有相关的文献吗？）This situation poses unique challenges as it requires the new agent or the intruder to adapt to the pattern of the original system. （本文的应用场景）Specifically, in our application scenario, a formation is initially operating normally when one of its nodes fails and exits the formation. At this point, an intruder is able to infiltrate the formation and replace the failed node. Therefore, addressing the formation intrusion problem is essential for enhancing the flexibility and adaptability of multi-agent systems.

In this paper, we propose a novel formation intrusion method based on complex Laplacian matrix control. The complex Laplacian matrix has been widely used in the analysis and control of multi-agent systems due to its ability to capture the topological structure and interaction dynamics of the system.（那两篇文献） By leveraging the properties of the complex Laplacian matrix, we model the formation intrusion system as a first-order linear time-invariant (LTI) system. This modeling approach allows us to systematically analyze the system behavior and develop effective control strategies.

To achieve formation intrusion, we first utilize the trajectory information of the formation system to predict the system parameters using an Ordinary Least Squares (OLS) estimator. （文献）Based on the predicted parameters, we derive the control law that enables the intruding agent to join the existing formation smoothly and maintain the desired formation structure. Moreover, we conduct a theoretical analysis of the convergence properties of the system parameter prediction results, providing a solid theoretical foundation for the proposed method.

The main contributions of this paper are summarized as follows:

1. We propose a novel formation intrusion method based on complex Laplacian matrix control, which is the first of its kind to address the formation intrusion problem in this manner.
2. We provide a comprehensive theoretical analysis of the convergence properties of the system parameter prediction results, ensuring the reliability and effectiveness of the proposed control strategy.
3. Through extensive simulation experiments, we validate the effectiveness of the proposed method and compare its performance with that of traditional formation control approaches.

The remainder of this paper is organized as follows. Section II presents the preliminaries and problem formulation. Section III introduces the method of using the Ordinary Least Squares (OLS) estimator to identify the system parameters. Section IV provides the theoretical analysis of the parameter prediction convergence. Simulation results are presented in Section V to demonstrate the effectiveness of the proposed method. Finally, conclusions are drawn in Section VI.



# Notations

在本文中，将编队中 Leaders 的序号放置在编队的最后，假设leaders的个数为 l，followers的个数为 f，n = l+f，所以followers的位置可以表示为pf = p[：f] = [p1,p2,...pf]，leaders的位置可以表示为pl = p[n-l:] = [p_{n-l+1},..., pn]

## Graph theory and complex Laplacian

定义 𝐺 = (V, E) 是无向图，该无向图包含一组非空的节点集合 V = 1, 2, . . . , 𝑛， 表示该无向图中共 𝑛 个节点，并且该无向图中边的集合表示为 E ⊆ V × V，即集合 E 中的一条边为无向图 𝐺 一对无序的节点对。在图 𝐺 = (V, E) 中，对于节点 𝑖 ∈ V，定义 N𝑖 为节点 i 的邻居节点的集合。

下面定义双向图，无向图是一种特殊的双向图，即无向图 𝐺 中两个节点 𝑖, 𝑗 之间的一条边可以看作两条方向相反的有向边 (𝑖, 𝑗) 与 ( 𝑗, 𝑖)，又可以称之为双向图。对于双向图，每一对节 点 𝑖, 𝑗 之间的两条边上的权重系数是可以不相同的。假设定义 节点 i 到节点 j 之间边的权重为 𝑤𝑖j。对于双向图，𝑤𝑖j不一定等于𝑤j𝑖

下面定义双向图的复拉普拉斯矩阵 $L$：
$$
L(i, j) = 
\begin{cases} 
-w_{ij} & \text{if } i \neq j \text{ and } j \in \mathcal{N}_i \\
0 & \text{if } i \neq j \text{ and } j \notin \mathcal{N}_i \\
\sum_{k \in \mathcal{N}_i} w_{ik} & \text{if } i = j
\end{cases}
$$


其中 w𝑖𝑗 ∈ 𝐶，𝑖, 𝑗 ∈ V

所以，可以进而表示 Lff = L[:f, :f]，Lfl = L[1:f, n-l+1:n]，Llf=L[n-l+1:n, 1:f], Lll = L[n-l+1:n, n-l+1:n]

当使用复拉普拉斯矩阵进行编队控制，实现编队控制平移、旋转、放缩过程中形状不变，有以下约束条件：1. Lp = 0，其中 L 是编队双向图的复拉普拉斯矩阵，p是编队所有节点的位置，$p = [p_1, p_2,...p_n] \in C^n$；2. 编队的拓扑结构是双根的









翻译：

```latex
Define an undirected graph $\mathcal{G} = (V, E)$, where $V = \{1, 2, \ldots, n\}$ represents a non-empty set of vertices, indicating that there are $n$ vertices in the graph $\mathcal{G}$. The set of edges $E \subseteq V \times V$ represents the unordered pairs of vertices in the graph. For a vertex $i \in V$, let $N_i$ denote the set of neighboring vertices of $i$.

A bidirectional graph is defined as a special type of undirected graph, where an edge between vertices $i$ and $j$ in the undirected graph $\mathcal{G}$ can be considered as two directed edges $(i, j)$ and $(j, i)$ with potentially different weights. Let the weight of the edge from vertex $i$ to vertex $j$ be denoted by $w_{ij}$. In a bidirectional graph, $w_{ij}$ is not necessarily equal to $w_{ji}$.

The complex Laplacian matrix $L$ of the bidirectional graph is defined as:
\begin{equation}
L(i, j) = 
\begin{cases} 
-w_{ij} & \text{if } i \neq j \text{ and } j \in N_i \\\
0 & \text{if } i \neq j \text{ and } j \notin N_i \\\
\sum_{k \in N_i} w_{ik} & \text{if } i = j
\end{cases}
\end{equation}
where $w_{ij} \in \mathbb{C}$ and $i, j \in V$.
```









































