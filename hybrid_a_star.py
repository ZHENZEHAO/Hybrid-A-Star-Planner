"""

Hybrid A* path planning

author: Zheng Zh (@Zhengzh)

"""

import heapq
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import sys
import pathlib 
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from dynamic_programming_heuristic import calc_distance_heuristic    #高级算法需要跑完代码，理解智能驾驶代码之后，深度学习此算法
import reeds_shepp_path_planning as rs           #同理
from car import move, check_car_collision, MAX_STEER, WB, plot_car, BUBBLE_R    #同理
 
XY_GRID_RESOLUTION = 2.0  # [m]
YAW_GRID_RESOLUTION = np.deg2rad(15.0)  # [rad]
MOTION_RESOLUTION = 0.1  # [m] path interpolate resolution
N_STEER = 20  # number of steer command

SB_COST = 100.0  # switch back penalty cost  #原数值是100       #Cost Function (代价函数)需要学习，惩罚系数各自为什么要这样定？原理是什么？；Heuristic Cost （启发函数）需要学习。
BACK_COST =5.0  # backward penalty cost #原数值是5
STEER_CHANGE_COST =5.0  # steer angle change penalty cost
STEER_COST = 1.0  # steer angle change penalty cost
H_COST = 5.0  # Heuristic cost

show_animation = True   #控制演示动画是否开启


class Node:              #class: 相当于创造一个档案袋（Node档案袋),需要使用提到Node即可使用，方便，快捷。

    def __init__(self, x_ind, y_ind, yaw_ind, direction,         #def:简化算法，若是相同的计算方法，只需要输入数值即可。简化算法。
                 x_list, y_list, yaw_list, directions,
                 steer=0.0, parent_index=None, cost=None):
        self.x_index = x_ind
        self.y_index = y_ind
        self.yaw_index = yaw_ind
        self.direction = direction
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.directions = directions
        self.steer = steer
        self.parent_index = parent_index
        self.cost = cost    #代价f=g+h:需要学习


class Path:

    def __init__(self, x_list, y_list, yaw_list, direction_list, cost):
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.direction_list = direction_list
        self.cost = cost


class Config:

    def __init__(self, ox, oy, xy_resolution, yaw_resolution):
        min_x_m = min(ox)    #设置地图边界
        min_y_m = min(oy)
        max_x_m = max(ox)
        max_y_m = max(oy)

        ox.append(min_x_m)   #确保地图边界被锁死（需要进一步学习）
        oy.append(min_y_m)
        ox.append(max_x_m)
        oy.append(max_y_m)

        self.min_x = round(min_x_m / xy_resolution)      #坐标离散化
        self.min_y = round(min_y_m / xy_resolution)
        self.max_x = round(max_x_m / xy_resolution)
        self.max_y = round(max_y_m / xy_resolution)

        self.x_w = round(self.max_x - self.min_x)       #确定网格数量
        self.y_w = round(self.max_y - self.min_y)

        self.min_yaw = round(- math.pi / yaw_resolution) - 1      #角度离散化
        self.max_yaw = round(math.pi / yaw_resolution)
        self.yaw_w = round(self.max_yaw - self.min_yaw)     #确定角度数量


def calc_motion_inputs(): #确定离散化后反方向盘转动位置和挡位配合的所有可能性
    for steer in np.concatenate((np.linspace(-MAX_STEER, MAX_STEER,     #np的函数计算需要学习
                                             N_STEER), [0.0])):
        for d in [1, -1]:
            yield [steer, d]    #列出所有可能性，具体什么形式来列出呢？（列表？还是？） 需要进一步学习


def get_neighbors(current, config, ox, oy, kd_tree):    #寻找下一步的所有安全路径
    for steer, d in calc_motion_inputs():
        node = calc_next_node(current, steer, d, config, ox, oy, kd_tree)
        if node and verify_index(node, config):  #verify_index 自定义函数后续需要学习
            yield node


def calc_next_node(current, steer, direction, config, ox, oy, kd_tree): #根据当前的位置和角度状态，找到下一步的位置和花费
    x, y, yaw = current.x_list[-1], current.y_list[-1], current.yaw_list[-1]

    arc_l = XY_GRID_RESOLUTION * 1.5
    x_list, y_list, yaw_list, direction_list = [], [], [], []
    for _ in np.arange(0, arc_l, MOTION_RESOLUTION):
        x, y, yaw = move(x, y, yaw, MOTION_RESOLUTION * direction, steer)  #move函数需要学习原理
        x_list.append(x)
        y_list.append(y)
        yaw_list.append(yaw)
        direction_list.append(direction == 1)

    if not check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree):  #碰撞检测函数需要学习原理
        return None
        

    d = direction == 1
    x_ind = round(x / XY_GRID_RESOLUTION)
    y_ind = round(y / XY_GRID_RESOLUTION)
    yaw_ind = round(yaw / YAW_GRID_RESOLUTION)

    added_cost = 0.0

    if d != current.direction:
        added_cost += SB_COST

    # steer penalty
    added_cost += STEER_COST * abs(steer)      #惩罚机制的原理，数值设置需要学习

    # steer change penalty
    added_cost += STEER_CHANGE_COST * abs(current.steer - steer)

    cost = current.cost + added_cost + arc_l

    node = Node(x_ind, y_ind, yaw_ind, d, x_list,
                y_list, yaw_list, direction_list,
                parent_index=calc_index(current, config),
                cost=cost, steer=steer)

    return node


def is_same_grid(n1, n2):       #检测两个坐标是否在同一位置  #\：换行符，为了美观，方便观看
    if n1.x_index == n2.x_index \
          and n1.y_index == n2.y_index \
          and n1.yaw_index == n2.yaw_index:
        return True
    return False


def analytic_expansion(current, goal, ox, oy, kd_tree):    #寻找两个位置之间的最短路径且最便宜的路径（cost)
    start_x = current.x_list[-1]
    start_y = current.y_list[-1]
    start_yaw = current.yaw_list[-1]

    goal_x = goal.x_list[-1]
    goal_y = goal.y_list[-1]
    goal_yaw = goal.yaw_list[-1]

    max_curvature = math.tan(MAX_STEER) / WB   #计算最大曲率，曲率公式应用需要学习
    paths = rs.calc_paths(start_x, start_y, start_yaw,   #rs.calc_paths代码的运算原理需要学习
                          goal_x, goal_y, goal_yaw,
                          max_curvature, step_size=MOTION_RESOLUTION)

    if not paths:
        return None

    best_path, best = None, None

    for path in paths:
        if check_car_collision(path.x, path.y, path.yaw, ox, oy, kd_tree):
            cost = calc_rs_path_cost(path)  #计算惩罚代价
            if not best or best > cost:
                best = cost
                best_path = path

    return best_path #需要了解path里面究竟有什么参数？


def update_node_with_analytic_expansion(current, goal,
                                        c, ox, oy, kd_tree):
    path = analytic_expansion(current, goal, ox, oy, kd_tree)

    if path:
        if show_animation:
            plt.plot(path.x, path.y)
        f_x = path.x[1:]
        f_y = path.y[1:]
        f_yaw = path.yaw[1:]

        f_cost = current.cost + calc_rs_path_cost(path)
        f_parent_index = calc_index(current, c)  #回溯整段路径需要学习

        fd = []
        for d in path.directions[1:]:
            fd.append(d >= 0)

        f_steer = 0.0
        f_path = Node(current.x_index, current.y_index, current.yaw_index,
                      current.direction, f_x, f_y, f_yaw, fd,
                      cost=f_cost, parent_index=f_parent_index, steer=f_steer)
        return True, f_path

    return False, None


def calc_rs_path_cost(reed_shepp_path):   #惩罚机制为何这样设定，设定原理需要深入学习  
    cost = 0.0
    for length in reed_shepp_path.lengths:    #reed_shepp_path自定义函数包含：lengths,ctypes(这个列表里包含： L S R),x, y, yaw(坐标)
        if length >= 0:  # forward
            cost += length
        else:  # back
            cost += abs(length) * BACK_COST

    # switch back penalty
    for i in range(len(reed_shepp_path.lengths) - 1):  #len:统计列表里元素数量   range:生成一个从零开始的列表 example: range(3) print(range) ：  [0,1,2]
        # switch back
        if reed_shepp_path.lengths[i] * reed_shepp_path.lengths[i + 1] < 0.0:  #用来引入序列号i看reed_shepp_path.lengths相邻两次行驶是否换挡位（前进挡和倒挡）
            cost += SB_COST

    # steer penalty
    for course_type in reed_shepp_path.ctypes:
        if course_type != "S":  # curve
            cost += STEER_COST * abs(MAX_STEER)

    # ==steer change penalty
    # calc steer profile
    n_ctypes = len(reed_shepp_path.ctypes)
    u_list = [0.0] * n_ctypes
    for i in range(n_ctypes):
        if reed_shepp_path.ctypes[i] == "R":
            u_list[i] = - MAX_STEER
        elif reed_shepp_path.ctypes[i] == "L":
            u_list[i] = MAX_STEER

    for i in range(len(reed_shepp_path.ctypes) - 1):
        cost += STEER_CHANGE_COST * abs(u_list[i + 1] - u_list[i])

    return cost


def hybrid_a_star_planning(start, goal, ox, oy, xy_resolution, yaw_resolution):
    """
    start: start node
    goal: goal node
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    xy_resolution: grid resolution [m]
    yaw_resolution: yaw angle resolution [rad]
    """

    start[2], goal[2] = rs.pi_2_pi(start[2]), rs.pi_2_pi(goal[2])   #将角度标准化，都计算到（-pi，pi）之间
    tox, toy = ox[:], oy[:]

    obstacle_kd_tree = cKDTree(np.vstack((tox, toy)).T)     #这句代码的含义是什么，怎么应用

    config = Config(tox, toy, xy_resolution, yaw_resolution)    #它输出了一些什么参数

    start_node = Node(round(start[0] / xy_resolution),    #start_node;goal_node自定义函数中Node函数没有定义：steer，parent_index.需要理解原理
                      round(start[1] / xy_resolution),
                      round(start[2] / yaw_resolution), True,
                      [start[0]], [start[1]], [start[2]], [True], cost=0)
    goal_node = Node(round(goal[0] / xy_resolution),
                     round(goal[1] / xy_resolution),
                     round(goal[2] / yaw_resolution), True,
                     [goal[0]], [goal[1]], [goal[2]], [True]) 

    openList, closedList = {}, {}   #{}的索引时间巨快，几乎为零   []列表:在寻找对象时从前往后，逐一寻找，计算量大，查找时间慢。

    h_dp = calc_distance_heuristic(       #1.一张全图的“距离终点距离表”（避障版），在出发前一次性算好（预处理），为了让后续的实时搜索过程快如闪电  2.深度学习此定义函数的编写原理
        goal_node.x_list[-1], goal_node.y_list[-1],
        ox, oy, xy_resolution, BUBBLE_R)

    pq = []
    openList[calc_index(start_node, config)] = start_node   #calc_index(start_node, config) 此定义函数是什么意思？怎么用？都输出了什么参数？
    heapq.heappush(pq, (calc_cost(start_node, h_dp, config),  #calc_cost此定义函数是什么意思？怎么用？都输出了什么参数？
                        calc_index(start_node, config)))
    final_path = None

    while True:
        if not openList:
            print("Error: Cannot find path, No open set")
            return Path([], [], [], [], 0)

        cost, c_id = heapq.heappop(pq)   
        if c_id in openList:
            current = openList.pop(c_id)
            closedList[c_id] = current
        else:
            continue
        
        #dist_to_goal=math.hypot(current.x_list[-1]-goal[0],current.y_list[-1]-goal[1])
        #grid_length=1.5
        #if dist_to_goal <= grid_length :
         #print("find final path")
         #final_path=current
        # break
            
        

        if show_animation:  # pragma: no cover
            plt.plot(current.x_list[-1], current.y_list[-1], "xc")
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',  #字符串常量，Matplotlib 库内部预定义好的“暗号”，其中还有：'key_press_event'：按键按下的瞬间（还没松开就触发）。'button_press_event'：鼠标点击。
                lambda event: [exit(0) if event.key == 'escape' else None])    #'motion_notify_event'：鼠标移动
            if len(closedList.keys()) % 10 == 0:  #每走十步刷新一下画面，在帧率和运算速度上面做了权衡，既能让用户看清有可以保证计算量不会过于大
                plt.pause(0.001)  #暂停0.001秒的同时刷新最新画面

        is_updated, final_path = update_node_with_analytic_expansion(
           current, goal_node, config, ox, oy, obstacle_kd_tree)

        if is_updated:
            print("path found")
            break

        for neighbor in get_neighbors(current, config, ox, oy,
                                      obstacle_kd_tree):
            neighbor_index = calc_index(neighbor, config)
            if neighbor_index in closedList:
                continue
            if neighbor_index not in openList \
                    or openList[neighbor_index].cost > neighbor.cost:
                heapq.heappush(
                    pq, (calc_cost(neighbor, h_dp, config),
                         neighbor_index))
                openList[neighbor_index] = neighbor
    

    path = get_final_path(closedList, final_path)  #输出了从起点到终点的路径，和花费
    return path


def calc_cost(n, h_dp, c):
    ind = (n.y_index - c.min_y) * c.x_w + (n.x_index - c.min_x)   #将二维数组（x,y）转化成一维，在字典中索引速度极快
    if ind not in h_dp:
        return n.cost + 999999999  # collision cost
    return n.cost + H_COST * h_dp[ind].cost


def get_final_path(closed, goal_node):  #找到起点到终点的路径
    reversed_x, reversed_y, reversed_yaw = \
        list(reversed(goal_node.x_list)), list(reversed(goal_node.y_list)), \
        list(reversed(goal_node.yaw_list))
    direction = list(reversed(goal_node.directions))
    nid = goal_node.parent_index    #nid 输出了什么，为什么可以索引下面closed
    final_cost = goal_node.cost

    while nid:
        n = closed[nid]
        reversed_x.extend(list(reversed(n.x_list)))
        reversed_y.extend(list(reversed(n.y_list)))
        reversed_yaw.extend(list(reversed(n.yaw_list)))
        direction.extend(list(reversed(n.directions)))

        nid = n.parent_index

    reversed_x = list(reversed(reversed_x))
    reversed_y = list(reversed(reversed_y))
    reversed_yaw = list(reversed(reversed_yaw))
    direction = list(reversed(direction))

    # adjust first direction
    direction[0] = direction[1]

    path = Path(reversed_x, reversed_y, reversed_yaw, direction, final_cost)

    return path


def verify_index(node, c):   #检测位置是否在安全范围之内
    x_ind, y_ind = node.x_index, node.y_index
    if c.min_x <= x_ind <= c.max_x and c.min_y <= y_ind <= c.max_y:
        return True

    return False


def calc_index(node, c):   #将三维数组（x,y,yaw）转化成一维，在字典中索引速度极快
    ind = (node.yaw_index - c.min_yaw) * c.x_w * c.y_w + \
          (node.y_index - c.min_y) * c.x_w + (node.x_index - c.min_x)

    if ind <= 0:
        print("Error(calc_index):", ind)

    return ind


def main():
    print("Start Hybrid A* planning")

    ox, oy = [], []

    for i in range(60):
        ox.append(i)
        oy.append(0.0)
    for i in range(40):
        ox.append(60.0)
        oy.append(i)
    for i in range(61):
        ox.append(i)
        oy.append(40.0)
    for i in range(40):
        ox.append(0.0)
        oy.append(i)

    for i in range(15, 46):
        ox.append(i)
        oy.append(15.0)
    
    for i in range(15, 46):
        ox.append(i)
        oy.append(25.0)
   
    for i in range(15, 26):
        ox.append(15.0)
        oy.append(i)
    for i in range(15, 26):
        ox.append(45.0)
        oy.append(i)
    start = [15.0, 32.0, np.deg2rad(180.0)] 
    
   
    goal = [55.0, 32.0, np.deg2rad(180.0)]

    print("start : ", start)
    print("goal : ", goal)

    if show_animation:
        plt.plot(ox, oy, ".k")
        rs.plot_arrow(start[0], start[1], start[2], fc='g')  #自定义函数需要学习
        rs.plot_arrow(goal[0], goal[1], goal[2])

        plt.grid(True)
        plt.axis("equal")
       

    path = hybrid_a_star_planning(
        start, goal, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)

    x = path.x_list
    y = path.y_list
    yaw = path.yaw_list

    if show_animation:
        for i_x, i_y, i_yaw in zip(x, y, yaw):
            plt.cla()  #清除坐标轴防止重影
            plt.plot(ox, oy, ".k")   #绘图，".k"： .表示用点来显示数据（障碍物）；k表示颜色使用黑色
            plt.plot(x, y, "-r", label="Hybrid A* path") #画出x,y。 "-r"：表示用线和红色表示数据
            plt.grid(True) #绘制网格线
            plt.axis("equal") #强制设置 X 轴和 Y 轴的 比例尺 (Aspect Ratio) 为 1:1
            plot_car(i_x, i_y, i_yaw) #自定义函数需要学习
            plt.pause(0.0001)

    print(__file__ + " done!!")


if __name__ == '__main__':
    main()
