import sys
sys.path.append('/Users/zhangyichi/opt/anaconda3/lib/python3.9/site-packages')

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

df=pd.read_excel('/Users/zhangyichi/Downloads/refined_depth_data.xlsx')
df=df.iloc[1:,:]

step_size=50

class SeabedMappingEnvironment:

    def __init__(self, depth_data, scan_angle, scan_depth, line_angle):
        """
        使用深度数据和扫描参数初始化环境。

        参数:
        - depth_data: 表示海床深度数据的2D数组。
        - scan_angle: 扫描波束的角度。
        - scan_depth: 扫描波束的深度。
        - line_angle: 被扫描线的角度。
        """
        self.depth_data = depth_data
        self.scan_angle = scan_angle
        self.scan_depth = scan_depth
        self.line_angle = line_angle
        self.position = (0, 0)
        self.scan_area = np.zeros_like(self.depth_data, dtype=bool)
        self.path = []

        # 初始化其他状态变量（如位置、扫描区域等）
        self.reset()

    def render(self):
        plt.imshow(self.depth_data)
        plt.plot(self.scan_area, 'r-')
        plt.scatter([self.position[1]], [self.position[0]], s=120, facecolors='none', edgecolors='b')
        plt.xlim([0, self.depth_data.shape[1]])
        plt.ylim([self.depth_data.shape[0], 0])
        plt.draw()
        plt.pause(0.001)
        plt.clf()

    def compute_scan_area(self, position, direction):
        # Step 1: 确定扫描区域的边界
        # 根据当前的扫描参数和代理的位置/方向来确定新的扫描区域的边界
        # 使用数学方程来找到两个对称平面的方程

        # 获取扫描角的一半（以弧度为单位）
        half_scan_angle_rad = np.radians(self.scan_angle / 2)

        # 计算两个对称平面的法向量
        normal_vector1 = [np.cos(direction + half_scan_angle_rad), np.sin(direction + half_scan_angle_rad)]
        normal_vector2 = [np.cos(direction - half_scan_angle_rad), np.sin(direction - half_scan_angle_rad)]

        # Step 2: 检查每个点是否在扫描区域内
        for i in range(self.depth_data.shape[0]):
            for j in range(self.depth_data.shape[1]):
                # 计算当前点的坐标
                x = i * self.resolution
                y = j * self.resolution

                # 检查点是否在两个对称平面定义的区域内
                in_scan_area = self.check_point_in_scan_area(x, y, position, normal_vector1, normal_vector2)

                # Step 3: 标记在扫描区域内的点
                if in_scan_area:
                    self.scan_area[i, j] = True

        # 计算新扫描区域
        new_scan_area = self.scan_area.copy()

        return new_scan_area

    def calculate_new_position(current_position, direction, step_size):
        """
        计算代理的新位置。

        参数：
        - current_position: 代理的当前位置，一个二元列表 [row, col]。
        - direction: 移动的方向，一个整数，代表移动方向。
        - step_size: 移动的步长，一个整数，代表移动距离。

        返回值：
        一个二元列表 [new_row, new_col]，代表计算得到的新位置。
        """
        # 定义移动方向的偏移量
        direction_offsets = {
            0: (0, 1),  # 向右移动
            1: (1, 1),  # 右上
            2: (1, 0),  # 上
            3: (1, -1),  # 左上
            4: (0, -1),  # 向左移动
            5: (-1, -1),  # 左下
            6: (-1, 0),  # 下
            7: (-1, 1)  # 右下
        }

        # 获取移动方向的偏移量
        offset = direction_offsets[direction]


        # 计算新位置
        new_row = current_position[0] + offset[0] * step_size
        new_col = current_position[1] + offset[1] * step_size

        return [new_row, new_col]

    def check_point_in_scan_area(self, x, y, position, normal_vector1, normal_vector2):
        # 计算点相对于当前位置的坐标
        relative_x = x - position[0]
        relative_y = y - position[1]

        # 计算点到两个平面的距离
        distance_to_plane1 = relative_x * normal_vector1[0] + relative_y * normal_vector1[1]
        distance_to_plane2 = relative_x * normal_vector2[0] + relative_y * normal_vector2[1]

        # 检查点是否在两个平面定义的区域内
        in_scan_area = distance_to_plane1 >= 0 and distance_to_plane2 <= 0

        return in_scan_area

    def reset(self):
        """
        将环境重置到其初始状态。
        """
        # 将状态变量重置为其初始值
        # （例如，位置回到起点，空扫描区域等）
        self.position = (0, 0)  # 将使用适当的起始位置更新此
        self.scan_area = np.zeros_like(self.depth_data, dtype=bool)  # 一个布尔数组表示扫描区域

    def step(self, action):
        """
        执行给定的动作并更新环境的状态。

        参数:
        - action: 要执行的动作。这将是一个包含移动方向和是否扫描的元组。

        返回:
        - reward: 从执行动作中获得的奖励。
        - done: 一个布尔值，指示情节是否完成。
        """
        # 提取动作的组件
        direction, scan = action

        # 定义24个可能的移动方向（每个方向间隔15度）
        directions = np.linspace(0, 2 * np.pi, 24, endpoint=False)

        # 根据选择的方向和一个固定的步长来更新代理的当前位置
        step_size = 1  # 我们可以稍后调整这个值
        dx = step_size * np.cos(directions[direction])
        dy = step_size * np.sin(directions[direction])
        self.position = (self.position[0] + dx, self.position[1] + dy)

        # 更新路径
        self.path.append(self.position)
        done = False
        return reward, done

    def compute_reward(self, new_scan_area):
        """
        给定一个新的扫描区域计算奖励。

        参数:
        - new_scan_area: 一个布尔数组表示新扫描的区域。

        返回:
        - reward: 计算得出的奖励。
        """
        # 根据新的扫描区域计算奖励
        # ...

        # 目前，我们将返回一个占位符值
        reward = 0.0
        return reward


    def render(self):
        # 绘制路径
        path_x = [p[1] for p in self.path]
        path_y = [p[0] for p in self.path]
        plt.plot(path_x, path_y, 'g-', linewidth=2)


# 占位符深度数据用于测试 (稍后我们将用真实数据替换它)
depth_data_placeholder = np.zeros((100, 100))

# 使用占位符值测试环境初始化
env = SeabedMappingEnvironment(depth_data_placeholder, scan_angle=120, scan_depth=70, line_angle=0)


# 定义参数
num_episodes = 1000  # 总回合数
max_steps_per_episode = 100  # 每回合最大时间步数
epsilon = 0.1  # epsilon 贪心策略中的 epsilon
alpha = 0.1  # 学习率
gamma = 0.9  # 折扣因子

# 初始状态和位置
initial_state = "start"
initial_position = [0, 0]

# 扫描区域的坐标
scan_area = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]


# 计算新位置
def calculate_new_position(position, direction, step_size):
    row, col = position
    if direction == "up":
        return [row - step_size, col]
    elif direction == "down":
        return [row + step_size, col]
    elif direction == "left":
        return [row, col - step_size]
    elif direction == "right":
        return [row, col + step_size]


import math


def calculate_scan_area(agent_position, scan_angle, scan_range):
    """
    计算扫描区域的坐标。

    参数：
    - agent_position: 代理的当前位置，一个二元列表 [row, col]。
    - scan_angle: 扫描角度，以度为单位。
    - scan_range: 扫描范围，代表扫描的最大距离。

    返回值：
    一个列表，包含扫描区域的坐标，每个坐标以二元列表 [row, col] 表示。
    """
    scan_area = []

    # 将扫描角度转换为弧度
    scan_angle_rad = math.radians(scan_angle)

    # 计算扫描区域的坐标
    for distance in range(1, scan_range + 1):
        # 根据代理的当前位置、扫描角度和距离计算新位置
        new_row = int(agent_position[0] + distance * math.sin(scan_angle_rad))
        new_col = int(agent_position[1] + distance * math.cos(scan_angle_rad))

        # 将新位置添加到扫描区域中
        scan_area.append([new_row, new_col])

    return scan_area

# 计算扫描操作的奖励
def calculate_scan_reward(scan_area):
    """
    计算扫描操作的奖励。

    参数：
    - scan_area: 扫描区域的坐标，一个包含二元列表 [row, col] 的列表。

    返回值：
    一个奖励值，表示扫描操作的奖励。
    """
    # 示例奖励计算：根据扫描覆盖的区域计算奖励
    reward = len(scan_area) * 0.1  # 扫描区域的坐标数量乘以奖励系数

    return reward


def calculate_expected_rewards(agent, q_values, scan_area):
    """
    计算每个可行动作的预期奖励。

    参数：
    - agent: 代理对象，包含代理的位置信息和已知信息。
    - q_values: 一个字典，包含每个方向的 Q 值（奖励）。
    - scan_area: 扫描区域的坐标，一个包含二元列表 [row, col] 的列表。

    返回值：
    一个字典，包含每个可行动作的预期奖励。
    """
    expected_rewards = {}

    # 遍历每个可行动作（移动和扫描）
    for action in ["move", "scan"]:
        if action == "move":
            # 如果是移动，计算每个方向的预期奖励
            for direction in q_values.keys():
                # 选择移动方向
                new_direction = direction
                # 计算新位置
                new_position = calculate_new_position(agent['position'], new_direction, step_size)
                # 计算预期奖励
                expected_rewards[(action, new_direction)] = q_values[direction]
        elif action == "scan":
            # 如果是扫描，计算扫描操作的奖励
            scan_reward = calculate_scan_reward(scan_area)
            # 预期奖励等于扫描操作的奖励
            expected_rewards[(action, None)] = scan_reward

    return expected_rewards


def choose_action(agent, epsilon, expected_rewards):
    """
    根据 epsilon 贪心策略选择动作。

    参数：
    - agent: 代理对象，包含代理的位置信息和已知信息。
    - epsilon: 探索概率，介于 0 和 1 之间。
    - expected_rewards: 每个可行动作的预期奖励，一个字典。

    返回值：
    所选动作（字符串），可能是 "move" 或 "scan"。
    """
    if random.random() < epsilon:
        # 以 epsilon 的概率随机选择动作
        return random.choice(["move", "scan"])
    else:
        # 以概率 1-epsilon 选择具有最高预期奖励的动作
        best_action = max(expected_rewards, key=lambda k: expected_rewards[k])
        return best_action[0]  # 返回动作名称，如 "move" 或 "scan"


def perform_action(agent, action, scan_area):
    """
    执行代理的动作，并计算奖励。

    参数：
    - agent: 代理对象，包含代理的位置信息和已知信息。
    - action: 所选动作，可能是 "move" 或 "scan"。
    - scan_area: 扫描区域的坐标，一个包含二元列表 [row, col] 的列表。

    返回值：
    - reward: 执行动作后的奖励。
    - new_position: 动作执行后的新位置，一个二元列表 [row, col]。
    """
    if action == "move":
        # 更新代理的位置
        new_position = calculate_new_position(agent['position'], agent['direction'], step_size)
        # 计算奖励，示例中简化为 0.1
        reward = 0.1
    elif action == "scan":
        # 计算扫描操作的奖励
        reward = calculate_scan_reward(scan_area)
        # 扫描操作不改变代理的位置
        new_position = agent['position']

    return reward, new_position


def update_q_values(q_values, state, action, reward, new_state, alpha, gamma):
    """
    使用 Q 学习算法更新 Q 值。

    参数：
    - q_values: 包含每个状态和动作的 Q 值，一个字典。
    - state: 当前状态，一个标识代理状态的对象。
    - action: 执行的动作，可能是 "move" 或 "scan"。
    - reward: 执行动作后的奖励。
    - new_state: 动作执行后的新状态，一个标识代理状态的对象。
    - alpha: 学习速率，介于 0 和 1 之间。
    - gamma: 折扣因子，介于 0 和 1 之间。
    """
    # 计算最大 Q 值
    max_q_value = max(q_values.get((new_state, a), 0) for a in ["move", "scan"])

    # 更新 Q 值
    q_values[(state, action)] = (1 - alpha) * q_values.get((state, action), 0) + alpha * (reward + gamma * max_q_value)
    return q_values

def perform_action(agent, action, scan_area):
    """
    执行代理的动作，并计算奖励。

    参数：
    - agent: 代理对象，包含代理的位置信息和已知信息。
    - action: 所选动作，可能是 "move" 或 "scan"。
    - scan_area: 扫描区域的坐标，一个包含二元列表 [row, col] 的列表。

    返回值：
    - reward: 执行动作后的奖励。
    - new_position: 动作执行后的新位置，一个二元列表 [row, col]。
    """
    if action == "move":
        # 更新代理的位置
        new_position = calculate_new_position(agent['position'], agent['direction'], step_size)
        # 计算奖励，示例中简化为 0.1
        reward = 0.1
    elif action == "scan":
        # 计算扫描操作的奖励
        reward = calculate_scan_reward(scan_area)
        # 扫描操作不改变代理的位置
        new_position = agent['position']

    return reward, new_position

env = SeabedMappingEnvironment(depth_data_placeholder, scan_angle=120, scan_depth=700, line_angle=0)
agent = {
  'position': initial_position,
  'direction': 'up'
}
# 主循环
for episode in range(num_episodes):
    # 初始化环境
    env.reset()
    agent_state = initial_state
    agent_position = initial_position
    q_values={}
    print(type(q_values))
    for t in range(max_steps_per_episode):
        expected_rewards = calculate_expected_rewards(q_values, agent_state, scan_area)
        print(type(q_values))
        agent['position'] = new_position
        print(type(q_values))
        action = choose_action(expected_rewards, epsilon)
        print(type(q_values))
        reward, new_position = perform_action(agent, action, scan_area)
        print(type(q_values))
        q_values=update_q_values(q_values, agent_state, action, reward, agent_state, alpha, gamma)
        print(type(q_values))
    # 在每个回合结束后执行其他操作，例如记录学习进度或调整参数
