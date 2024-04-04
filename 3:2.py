import sys
sys.path.append('/Users/zhangyichi/opt/anaconda3/lib/python3.9/site-packages')

from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

# 设置参数和常量
D0 = 110
alpha = 1.5
theta = 120
x_min, x_max = 0, 4
y_min, y_max = 0, 2
x_center, y_center = 2, 1
conversion_factor = 1852
eta_target = 0.1
k = np.tan(np.deg2rad(theta) / 2)
tan_alpha = np.tan(np.deg2rad(alpha))

# 结果存储
results = []


def calculate_intersection_point(K, b):
    intersection_point = []
    if y_min <= b <= y_max:
        intersection_point.append((x_min, b))
    try:
        x = (y_max - b) / K
        if x_min < x < x_max:
            intersection_point.append((x, y_max))
    except:
        None

    try:
        y=(-b)/K
        if x_min < y < x_max:
            intersection_point.append((y, y_min))

    except:
        None

    if y_min <= 4 * K + b <= y_max:
        intersection_point.append((x_max, 4 * K + b))
    return intersection_point


def calculate_eta(DA, DB, d_double_prime, k, V_y, tan_alpha):
    term1 = k * (DA * (1 - k * V_y * tan_alpha) + DB * (1 + k * V_y * tan_alpha))
    term2 = d_double_prime * (1 - (k * V_y * tan_alpha) ** 2)
    eta = 1 / (2 * k * DA) * (term1 - term2)
    return eta


def objective(beta):
    if np.ndim(beta) > 0:
        beta = beta[0]

    V_y = np.sin(np.deg2rad(beta))
    K = -np.tan(np.deg2rad(beta))

    def total_line_length(d_double_prime, beta, V_y, K):
        x_current = x_min
        b_current = y_max
        total_length = 0

        while x_current < x_max:
            DA_current = D0 - (x_current - x_center) * tan_alpha * conversion_factor
            b_new = b_current - d_double_prime / np.sin(np.deg2rad(beta))
            intersection_points = calculate_intersection_point(K, b_new)

            if len(intersection_points) == 2:
                x_new = max(intersection_points[0][0], intersection_points[1][0])
                DB_new = D0 - (x_new - x_center) * tan_alpha * conversion_factor
                eta_new = calculate_eta(DA_current, DB_new, d_double_prime * conversion_factor, k, V_y, tan_alpha)

                if 0.1 <= eta_new <= 0.2:
                    total_length += np.linalg.norm(np.array(intersection_points[0]) - np.array(intersection_points[1]))
                    x_current = x_new
                    b_current = b_new
                else:
                    break
            else:
                break
        return total_length

    d_double_prime_values = np.arange(0.1, 1.01, 0.01)
    lengths = [total_line_length(d, beta, V_y, K) for d in d_double_prime_values]

    min_length = min(lengths)
    results.append((beta, min_length))

    return min_length


# 优化问题求解
res = minimize(objective, [91], bounds=[(91, 179)], method='SLSQP')
optimal_beta = res.x[0]
optimal_total_length = res.fun

# 输出最优解
print(f"Optimal beta: {optimal_beta:.2f}°")
print(f"Optimal total line length: {optimal_total_length:.2f} 海里")

# 可视化不同 beta 值的结果
betas, total_lengths = zip(*results)
plt.plot(betas, total_lengths)
plt.xlabel('Beta (°)')
plt.ylabel('Total line length (海里)')
plt.title('Total line length for different beta values')
plt.grid(True)
plt.show()

