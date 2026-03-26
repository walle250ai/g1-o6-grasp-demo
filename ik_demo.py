"""
IK 逆运动学演示 —— 平面二连杆机械臂
天树探界 · 机器人仿真抓取课程配套脚本

运行方式：python ik_demo.py
依赖：numpy, matplotlib（pip install numpy matplotlib）
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ── 中文字体自动检测 ────────────────────────────────
_chinese_fonts = [
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
]
for _fp in _chinese_fonts:
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
        _prop = font_manager.FontProperties(fname=_fp)
        matplotlib.rcParams['font.family'] = _prop.get_name()
        print(f"[字体] 使用：{_prop.get_name()}")
        break
else:
    print("[字体] 未找到中文字体，图中文字可能显示为方块，不影响运行")

matplotlib.rcParams['axes.unicode_minus'] = False

L1 = 1.0
L2 = 0.8

def forward_kinematics(theta1, theta2):
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)
    return (x1, y1), (x2, y2)

def inverse_kinematics(x, y):
    d = np.sqrt(x**2 + y**2)
    if d > L1 + L2:
        print(f"  ❌ 目标点 ({x:.2f}, {y:.2f}) 超出最大臂展 {L1+L2:.2f}m，不可达")
        return None
    if d < abs(L1 - L2):
        print(f"  ❌ 目标点 ({x:.2f}, {y:.2f}) 过近，不可达")
        return None
    cos_theta2 = np.clip((d**2 - L1**2 - L2**2) / (2 * L1 * L2), -1, 1)
    theta2 = -np.arccos(cos_theta2)
    k1 = L1 + L2 * np.cos(theta2)
    k2 = L2 * np.sin(theta2)
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    return theta1, theta2

def draw_arm(ax, theta1, theta2, target=None, title=""):
    ax.clear()
    (x1, y1), (x2, y2) = forward_kinematics(theta1, theta2)
    ax.plot([0, x1], [0, y1], 'o-', color='#2196F3', linewidth=6, markersize=10, label=f'大臂 L1={L1}m')
    ax.plot([x1, x2], [y1, y2], 'o-', color='#FF9800', linewidth=6, markersize=10, label=f'小臂 L2={L2}m')
    if target:
        ax.plot(*target, 'r*', markersize=18, label=f'目标 ({target[0]:.1f}, {target[1]:.1f})')
        ax.plot(*target, 'ro', markersize=22, alpha=0.25)
    ax.plot(x2, y2, 's', color='green', markersize=12, label=f'末端 ({x2:.2f}, {y2:.2f})')
    ax.add_patch(plt.Circle((0,0), L1+L2, color='gray', fill=False, linestyle='--', alpha=0.3))
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', alpha=0.2); ax.axvline(0, color='k', alpha=0.2)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(title, fontsize=10)

def main():
    targets = [(1.2,0.8),(0.5,1.3),(-1.0,0.6),(1.8,0.0),(2.5,0.0)]
    print("="*50)
    print(f"  逆运动学演示  大臂L1={L1}m  小臂L2={L2}m  最大臂展={L1+L2}m")
    print("="*50)

    fig, axes = plt.subplots(1, len(targets), figsize=(18, 4))
    fig.suptitle("IK 逆运动学演示：给定末端目标位置 -> 自动求解关节角", fontsize=13, fontweight='bold')

    for i, (tx, ty) in enumerate(targets):
        print(f"\n目标点 ({tx:5.1f}, {ty:5.1f})：")
        result = inverse_kinematics(tx, ty)
        if result:
            theta1, theta2 = result
            print(f"  theta1 = {np.degrees(theta1):7.2f} deg  (肩关节)")
            print(f"  theta2 = {np.degrees(theta2):7.2f} deg  (肘关节)")
            _, (ex, ey) = forward_kinematics(theta1, theta2)
            err = np.sqrt((ex-tx)**2+(ey-ty)**2)
            print(f"  验证误差 = {err:.8f}m")
            title = f"目标({tx},{ty})\nth1={np.degrees(theta1):.1f}  th2={np.degrees(theta2):.1f}"
            draw_arm(axes[i], theta1, theta2, target=(tx,ty), title=title)
        else:
            axes[i].set_facecolor('#fff0f0')
            axes[i].text(0, 0, f'目标点({tx},{ty})\n不可达', ha='center', va='center', fontsize=11, color='red')
            axes[i].set_xlim(-2.2,2.2); axes[i].set_ylim(-2.2,2.2)
            axes[i].set_aspect('equal'); axes[i].grid(True, alpha=0.3)
            axes[i].set_title(f"目标({tx},{ty})\n超出臂展", fontsize=10)

    print("\n" + "="*50)
    print("  关键结论：")
    print("  1. IK 本质：已知末端位置 -> 反推关节角")
    print("  2. 同一位置可能有多个解（肘朝上/朝下）")
    print("  3. 超出臂展不可达，IK 无解")
    print("  4. 真实机器人有力矩限制，IK 有解不代表能执行到位")
    print("="*50)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ik_demo_output.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n图片已保存到 {save_path}")
    plt.show()

if __name__ == '__main__':
    main()
