import os
import re
import matplotlib.pyplot as plt
import sys

import numpy as np

def mean_relative_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # 避免分母为零
    if np.any(y_true == 0):
        raise ValueError("真实值包含零，需处理以避免除零错误")
    # relative_errors = np.abs((y_true - y_pred) / y_true)
    relative_errors = 2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))
    return np.mean(relative_errors)


def parse_log_file(log_path):
    """解析日志文件，提取iteration和lm loss数据"""
    iterations = []
    lm_losses = []
    tgss = []
    seq_load_balancing_losses = []
    grad_norms = []
    
    # 匹配格式示例：iteration       85/   95367 | ... | lm loss: 9.924931E+00 | ...
    iteration_pattern = re.compile(
        r'iteration\s+(\d+)'
    )

    loss_pattern = re.compile(
        r"lm loss: \d+\.?\d*[eE][+-]?\d+"
    )

    tgs_pattern = re.compile(
        r"throughput per GPU \(TFLOP/s/GPU\): \d+\.?\d*"
    )

    seq_load_balancing_loss_pattern = re.compile(
        r"seq_load_balancing_loss: (\d+\.?\d*[eE][+-]?\d+)"
    )

    load_balancing_loss_pattern = re.compile(
        r"load_balancing_loss: (\d+\.?\d*[eE][+-]?\d+)"
    )

    grad_norm_pattern = re.compile(
        r"grad norm: \d+\.?\d*"
    )
    
    with open(log_path, 'r') as f:
        for line in f:
            if "lm loss:" not in line or "iteration" not in line:
                continue
            match = iteration_pattern.search(line)
            assert match
            iteration = int(match.group(1))
            iterations.append(iteration)

            match = loss_pattern.search(line)
            assert match
            lm_loss = float(match.group(0)[9:])
            lm_losses.append(lm_loss)

            match = tgs_pattern.search(line)
            assert match
            tgs = float(match.group(0).split(' ')[-1])
            tgss.append(tgs)

            # match = seq_load_balancing_loss_pattern.search(line)
            # if not match:
            #     match = load_balancing_loss_pattern.search(line)
            # assert match
            # seq_load_balancing_loss = float(match.group(0).split(' ')[-1])
            # seq_load_balancing_losses.append(seq_load_balancing_loss)
    
            match = grad_norm_pattern.search(line)
            assert match
            grad_norm = float(match.group(0).split(' ')[-1])
            grad_norms.append(grad_norm)

    return iterations, lm_losses, tgss, seq_load_balancing_losses, grad_norms

def plot_loss_curves(iterations1, losses1, label1, 
                     iterations2, losses2, label2, name='loss'):
    """绘制两条损失曲线的对比图"""
    plt.figure(figsize=(12, 6))
    
    # 绘制两条曲线
    plt.plot(iterations1, losses1, 'b-', label=label1, alpha=0.8)
    plt.plot(iterations2, losses2, 'r-', label=label2, alpha=0.8)
    
    # 添加标签和标题
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel(f'LM {name}', fontsize=12)
    plt.title(f'LM {name} Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 自动调整布局并显示
    plt.tight_layout()
    plt.savefig(f'{name}_comparison.png', dpi=100)


def plot_diff_curves(iterations, diff, label1, label2, name='loss'):
    """
    绘制两条曲线的差值图
    参数:
        iterations: 迭代次数列表（x轴）
        diff: 损失值差值列表（y轴）
        label1: 第一条曲线标签
        label2: 第二条曲线标签
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制差值曲线[8](@ref)
    plt.plot(iterations, diff, 'g-', label=f'Difference ({label1} - {label2})', alpha=0.8)
    
    # 添加零值参考线[3](@ref)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 标记最大/最小差值点[2](@ref)
    max_diff_idx = diff.index(max(diff))
    min_diff_idx = diff.index(min(diff))
    plt.scatter(iterations[max_diff_idx], diff[max_diff_idx], 
                color='purple', s=80, zorder=5, label='Max Difference')
    plt.scatter(iterations[min_diff_idx], diff[min_diff_idx], 
                color='orange', s=80, zorder=5, label='Min Difference')
    
    # 添加标签和标题
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel(f'{name} Difference', fontsize=12)
    plt.title(f'{name} Difference: {label1} vs {label2}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 自动调整布局并保存
    plt.tight_layout()
    plt.savefig(f'{name}_diff.png', dpi=100)
    plt.close()  # 关闭图形释放内存

  
def plot_prefix_mean_relative_error_curve(iterations, y_true, y_pred, name='lm_loss'):
    """
    绘制“前缀平均相对误差”随 align 变化的曲线。
    y_true 一般传 base，y_pred 一般传 exp。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) == 0:
        raise ValueError("输入为空，无法绘制前缀平均相对误差曲线")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true 和 y_pred 长度不一致")
    if np.any(y_true == 0):
        raise ValueError("真实值包含零，无法计算相对误差")

    # relative_errors = np.abs((y_true - y_pred) / y_true)
    relative_errors = 2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))
    prefix_mre = np.cumsum(relative_errors) / np.arange(1, len(relative_errors) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(iterations, prefix_mre, label='Prefix Mean Relative Error', alpha=0.8)
    plt.scatter(iterations[-1], prefix_mre[-1], s=80, zorder=5,
                label=f'Final MRE={prefix_mre[-1]:.6e}')

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Mean Relative Error', fontsize=12)
    plt.title(f'Prefix Mean Relative Error({name})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{name}_prefix_mean_relative_error.png', dpi=100)
    plt.close()

    return prefix_mre



if __name__ == "__main__":
    file2 = os.path.join(
        os.environ['MEGATRON_LM_MUSA_PATCH_HOME'],
        'examples/kimi-k2/output_log/2026-04-17_160621/tp1_pp4_dp16_mbs1_numbs16_gbs256_gpus64/7/none_008thaui/attempt_0/7/stdout.log',
    )
    file1 = os.path.join(
        os.environ['MEGATRON_LM_MUSA_PATCH_HOME'],
        'examples/kimi-k2/output_log/2026-04-22_185747/tp1_pp4_dp16_mbs1_numbs16_gbs256_gpus64/7/none_rs5tv4_n/attempt_0/7/stdout.log',
    )
    iterations1, losses1, tgs1, seq_load_balancing_losses1, grad_norms1 = parse_log_file(file1)
    iterations2, losses2, tgs2, seq_load_balancing_losses2, grad_norms2 = parse_log_file(file2)

    align = min(len(iterations1), len(iterations2))
    align = 8000
    iterations1 = iterations1[:align]
    iterations2 = iterations2[:align]
    losses1 = losses1[:align]
    losses2 = losses2[:align]
    grad_norms1 = grad_norms1[:align]
    grad_norms2 = grad_norms2[:align]


    diff = [a - b for a, b in zip(losses1, losses2)]
    plot_diff_curves(iterations1, diff, "exp", "base")

    print(f'avg_diff: {mean_relative_error(losses2, losses1)}')
    
    prefix_mre = plot_prefix_mean_relative_error_curve(
        iterations1,
        losses2,   # base / true
        losses1,   # exp / pred
        name='lm_loss'
    )

    print(f'final prefix avg diff: {prefix_mre[-1]}')

    if not iterations1 or not iterations2:
        print("Error: No valid lm loss data found in one or both files")
        sys.exit(1)
    
    plot_loss_curves(iterations1, losses1, "exp",
                     iterations2, losses2, "base")
    

    plot_loss_curves(iterations1, grad_norms1, "exp",
                     iterations2, grad_norms2, "base", 'grad_norms')

    

    avg_tgs1 = sum(tgs1[200:300]) / 100
    avg_tgs2 = sum(tgs2[200:300]) / 100


    print(f'exp avg tgs: {avg_tgs1}, base avg tgs: {avg_tgs2}')

