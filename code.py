import numpy as np
import matplotlib.pyplot as plt


class ThermalSolver:
    def __init__(self, Lx=15, Ly=12):
        self.Lx = Lx  # x方向物理尺寸(cm)
        self.Ly = Ly  # y方向物理尺寸(cm)

    def solve(self, m=50, n=40, omega=1.0, tol=1e-3, max_iter=5000):
        """求解二维稳态温度场"""
        T = np.zeros((m, n))

        # 设置边界条件
        T[:, -1] = 100  # 上边界
        T[0, :] = 20  # 左边界
        T[-1, :] = 20  # 右边界
        T[:, 0] = 20  # 下边界

        residuals = []
        for _ in range(max_iter):
            max_err = 0
            # 更新内部节点
            for i in range(1, m - 1):
                for j in range(1, n - 1):
                    old = T[i, j]
                    new = (T[i + 1, j] + T[i - 1, j] + T[i, j + 1] + T[i, j - 1]) / 4
                    T[i, j] = (1 - omega) * old + omega * new
                    max_err = max(max_err, abs(T[i, j] - old))

            residuals.append(max_err)
            if max_err < tol:
                break

        return T, residuals

# ================== 主程序 ==================

if __name__ == "__main__":
    solver = ThermalSolver()

    # 调整网格配置参数
    grid_configs = [
        {'m': 5, 'n': 4, 'label': '5x4 (Coarsest)', 'color': 'blue',
         'omega_range': (0.5, 1.6), 'tol': 1e-2},
        {'m': 15, 'n': 12, 'label': '15x12 (Medium)', 'color': 'green',
         'omega_range': (1.0, 1.8), 'tol': 1e-3},
        {'m': 30, 'n': 24, 'label': '30x24 (Fine)', 'color': 'red',
         'omega_range': (1.2, 1.9), 'tol': 1e-3},
        {'m': 50, 'n': 40, 'label': '50x40 (Finest)', 'color': 'purple',
         'omega_range': (1.5, 1.95), 'tol': 1e-3}
    ]

    plt.figure(figsize=(14, 8))

    for config in grid_configs:
        # 参数提取
        m = config['m']
        n = config['n']
        label = config['label']
        color = config['color']
        omega_start, omega_end = config['omega_range']
        tol = config['tol']  # 不同网格使用不同收敛标准

        # 生成omega范围
        omega_range = np.linspace(omega_start, omega_end, 30)

        iterations = []
        valid_omegas = []

        for omega in omega_range:
            # 调整求解参数
            _, res = solver.solve(m=m, n=n, omega=omega,
                                  tol=tol,  # 使用网格专属收敛标准
                                  max_iter=10000)

            # 收敛判断条件
            if len(res) < 10000 and res[-1] < tol:
                iterations.append(len(res))
                valid_omegas.append(omega)

        # 仅在有数据时绘图
        if len(valid_omegas) > 0:
            plt.semilogy(valid_omegas, iterations, 's-',
                         markersize=8,
                         linewidth=2,
                         color=color,
                         label=label)

            # 标注最佳点
            try:
                best_idx = np.argmin(iterations)
                plt.annotate(f'ω={valid_omegas[best_idx]:.2f}\n{iterations[best_idx]}',
                             (valid_omegas[best_idx], iterations[best_idx]),
                             textcoords="offset points",
                             xytext=(10, -20),
                             ha='left',
                             arrowprops=dict(arrowstyle="->", color=color),
                             color=color)
            except Exception as e:
                print(f"标注失败: {label} - {str(e)}")
        else:
            print(f"警告: {label} 无有效数据")

    # 图表参数调整
    plt.xlabel('Relaxation Factor ω', fontsize=12)
    plt.ylabel('Iterations (log scale)', fontsize=12)
    plt.title('Optimized Relaxation Factor Analysis', fontsize=14)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(loc='upper left')
    plt.xlim(0.4, 2.0)
    plt.ylim(5, 10000)
    plt.tight_layout()
    plt.savefig('convergence_analysis.jpg',
                dpi=300,
                bbox_inches='tight')
    plt.show()

    # 最密网格可视化
    plt.figure(figsize=(12, 8))
    T_fine, _ = solver.solve(m=50, n=40, omega=1.87)

    X, Y = np.meshgrid(np.linspace(0, 15, 50),
                       np.linspace(0, 12, 40))

    levels = np.arange(20, 101, 5)
    contour = plt.contour(X, Y, T_fine.T, levels=levels, colors='white', linewidths=0.5)
    plt.clabel(contour, inline=True, fmt='%d℃', fontsize=8)

    im = plt.imshow(T_fine.T, extent=[0, 15, 0, 12], origin='lower',
                    cmap='inferno', alpha=0.9)
    plt.colorbar(im, label='Temperature (°C)', shrink=0.8)

    plt.title('Temperature Field (50x40 Grid)', fontsize=14)
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.tight_layout()
    plt.savefig('temperature_field.jpg',
                dpi=300,
                bbox_inches='tight')
    plt.show()