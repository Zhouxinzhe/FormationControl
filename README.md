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



























































