import sys
sys.path.append('/Users/zhangyichi/opt/anaconda3/lib/python3.9/site-packages')

import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
import seaborn as sns

# 设置seaborn风格
sns.set_style("darkgrid")

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# 参数定义
D0 = 110  # 海域中心点处的海水深度，单位：m
alpha = 1.5  # 坡度，单位：度
theta = 120  # 多波束换能器的开角，单位：度

# 海域边界定义
x_min, x_max = 0, 4  # 单位：海里
y_min, y_max = 0, 2  # 单位：海里

# 海域中心点定义
x_center, y_center = 2, 1  # 单位：海里

# 海里到米的转换系数
conversion_factor = 1852

# 初始化测线列表
lines = []

# 初始化第一条测线（南北向边界线）
x_current = x_min
lines.append([(x_current, y_min), (x_current, y_max)])


# 计算η的函数
def calculate_eta(DA, DB, d_double_prime):
    k = np.tan(np.deg2rad(theta) / 2)
    V_y = 1  # 单位向量的y分量
    tan_alpha = np.tan(np.deg2rad(alpha))

    term1 = k * (DA * (1 - k * V_y * tan_alpha) + DB * (1 + k * V_y * tan_alpha))
    term2 = d_double_prime * (1 - (k * V_y * tan_alpha) ** 2)

    eta = 1 / (2 * k * DA) * np.abs(term1 - term2)
    return eta


# 生成后续的测线
eta_target = 0.1
x_current = x_min
line_count = 0

while x_current < x_max:
    # 计算当前点的深度
    DA_current = D0 - (x_current - x_center) * np.tan(np.deg2rad(alpha)) * conversion_factor


    # 定义方程求解新的测线
    def equation(d_double_prime):
        # 计算新的深度
        DB_new = D0 - (x_current + d_double_prime - x_center) * np.tan(np.deg2rad(alpha)) * conversion_factor

        # 计算η
        eta_new = calculate_eta(DA_current, DB_new, d_double_prime * conversion_factor)

        return eta_new - eta_target


    # 解方程找到d''
    res = root(equation, [0.1])
    d_double_prime_new = res.x[0]

    # 计算新的x坐标
    x_new = x_current + d_double_prime_new

    # 保证新的x值在边界内
    if x_new >= x_max:
        break

    # 添加新的测线
    lines.append([(x_new, y_min), (x_new, y_max)])

    # 更新当前的x值为新找到的x值
    x_current = x_new
    line_count += 1

# 绘制所有测线
plt.figure()
for i, line in enumerate(lines):
    plt.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]])
print(lines)
# 绘制海域边界
plt.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], 'k--')

# 设置图像属性
plt.xlabel('X (海里)')
plt.ylabel('Y (海里)')
plt.title('测线分布图')
plt.grid(True)

# 调整布局以适应图例
plt.tight_layout()

# 显示图像
plt.show()
