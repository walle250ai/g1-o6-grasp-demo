#!/bin/bash
# ============================================================
#  pack_for_students.sh
#  在教师机器上运行，生成学生可以直接解压使用的离线包
#
#  用法：bash pack_for_students.sh
#  输出：g1_grasp_demo_offline.tar.gz（约 4-6 GB）
# ============================================================
set -e

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$DEMO_DIR/manipulator_grasp"
OUTPUT_DIR="$DEMO_DIR/student_package"
OUTPUT_TAR="$DEMO_DIR/g1_grasp_demo_offline.tar.gz"
ENV_NAME="g1_graspnet"
# 自动检测 miniconda/anaconda 根目录
CONDA_ROOT="${CONDA_PREFIX_1:-${CONDA_PREFIX:-}}"
if [ -z "$CONDA_ROOT" ]; then
    # 尝试常见路径
    for p in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda" "/usr/local/conda"; do
        [ -d "$p/bin" ] && CONDA_ROOT="$p" && break
    done
fi
CONDA_PACK="$CONDA_ROOT/bin/conda-pack"
CONDA_BIN="$CONDA_ROOT/bin"

echo "=================================================="
echo "  G1 机器人抓取演示 — 学生离线包打包工具"
echo "=================================================="
echo ""
echo "项目路径: $PROJECT_DIR"
echo "输出文件: $OUTPUT_TAR"
echo ""

# ── 1. 检查依赖 ────────────────────────────────────────────
echo "[1/5] 检查环境..."
if [ ! -x "$CONDA_PACK" ]; then
    echo "错误: 未找到 conda-pack（查找路径: $CONDA_PACK）"
    echo "请安装: $CONDA_ROOT/bin/pip install conda-pack"
    exit 1
fi
if ! "$CONDA_BIN/conda" env list | grep -q "^$ENV_NAME "; then
    echo "错误: conda 环境 '$ENV_NAME' 不存在"
    exit 1
fi
echo "  ✓ conda-pack 已安装"
echo "  ✓ 环境 '$ENV_NAME' 已找到"

# ── 2. 清理旧打包目录 ──────────────────────────────────────
echo ""
echo "[2/5] 准备打包目录..."
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# ── 3. conda-pack 打包环境（最耗时的步骤）─────────────────
echo ""
echo "[3/5] 打包 Python 环境（这一步需要 10-30 分钟，请耐心等待）..."
echo "      正在压缩 $ENV_NAME 环境..."
"$CONDA_PACK" -n "$ENV_NAME" -o "$OUTPUT_DIR/env.tar.gz" --ignore-editable-packages
echo "  ✓ 环境打包完成：$(du -sh $OUTPUT_DIR/env.tar.gz | cut -f1)"

# ── 4. 复制项目代码 ────────────────────────────────────────
echo ""
echo "[4/5] 复制项目文件..."
rsync -a --progress \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='*.egg-info' \
    --exclude='.vscode' \
    --exclude='g1_graspnet_results*' \
    --exclude='MUJOCO_LOG.TXT' \
    --exclude='*.json' \
    --exclude='*.png' \
    --exclude='*.bak' \
    "$PROJECT_DIR/" "$OUTPUT_DIR/project/"
echo "  ✓ 项目文件复制完成：$(du -sh $OUTPUT_DIR/project | cut -f1)"

# ── 5. 生成学生用脚本 ─────────────────────────────────────
echo ""
echo "[5/5] 生成学生脚本..."

# install.sh ─────────────────────────────────────────────────
cat > "$OUTPUT_DIR/install.sh" << 'INSTALL_EOF'
#!/bin/bash
# ============================================================
#  install.sh —— 首次运行，解压 Python 环境（只需执行一次）
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$SCRIPT_DIR/env"
ENV_TAR="$SCRIPT_DIR/env.tar.gz"

echo "=================================================="
echo "  G1 机器人抓取演示 — 环境安装"
echo "=================================================="
echo ""

# 检查是否已经安装
if [ -f "$ENV_DIR/.installed" ]; then
    echo "✓ 环境已安装，无需重复安装。直接运行 ./run.sh 即可。"
    exit 0
fi

# 检查系统依赖
echo "[1/3] 检查系统依赖..."
MISSING=""
for pkg in libgl1 libglib2.0-0 libxrender1 libxext6; do
    if ! dpkg -l "$pkg" &>/dev/null 2>&1; then
        MISSING="$MISSING $pkg"
    fi
done
if [ -n "$MISSING" ]; then
    echo "  需要安装系统库（需要 sudo 密码）..."
    sudo apt-get install -y $MISSING
fi
echo "  ✓ 系统依赖就绪"

# 检查 NVIDIA 驱动
echo "[2/3] 检查 NVIDIA 驱动..."
if ! command -v nvidia-smi &>/dev/null; then
    echo ""
    echo "  ⚠ 警告：未检测到 NVIDIA 驱动！"
    echo "  GraspNet 需要 CUDA GPU，请确认：你的电脑有 NVIDIA 显卡且已安装驱动。"
    echo "  如果你使用云服务器，请选择带 GPU 的实例。"
    echo ""
    read -p "  是否继续安装？(y/N): " cont
    [[ "$cont" =~ ^[Yy]$ ]] || exit 1
else
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo "  ✓ 检测到 GPU: $GPU_INFO"
fi

# 解压 Python 环境
echo "[3/3] 解压 Python 环境（约需 3-8 分钟）..."
mkdir -p "$ENV_DIR"
tar -xzf "$ENV_TAR" -C "$ENV_DIR"
echo "  正在修复环境路径..."
source "$ENV_DIR/bin/activate"
conda-unpack
echo "  ✓ 环境解压完成"

# 标记已安装
touch "$ENV_DIR/.installed"

echo ""
echo "=================================================="
echo "  安装完成！运行演示："
echo "    ./run.sh"
echo "  参数说明："
echo "    ./run.sh --loops 5      # 循环 5 次"
echo "    ./run.sh --speed 1.0    # 1x 实时速度"
echo "    ./run.sh --step         # 逐步模式（课堂讲解）"
echo "=================================================="
INSTALL_EOF
chmod +x "$OUTPUT_DIR/install.sh"

# run.sh ─────────────────────────────────────────────────────
cat > "$OUTPUT_DIR/run.sh" << 'RUN_EOF'
#!/bin/bash
# ============================================================
#  run.sh —— 启动 G1 机器人抓取演示
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$SCRIPT_DIR/env"
PROJECT_DIR="$SCRIPT_DIR/project"

# 检查是否已安装
if [ ! -f "$ENV_DIR/.installed" ]; then
    echo "请先运行安装脚本: ./install.sh"
    exit 1
fi

# 激活环境并运行
source "$ENV_DIR/bin/activate"
cd "$PROJECT_DIR"
echo "启动 G1 机器人抓取演示..."
echo "（关闭 MuJoCo 窗口或按 Ctrl+C 退出）"
echo ""
python view_grasp_demo.py "$@"
RUN_EOF
chmod +x "$OUTPUT_DIR/run.sh"

# README ─────────────────────────────────────────────────────
cat > "$OUTPUT_DIR/README.txt" << 'README_EOF'
G1 机器人抓取演示 — 快速开始
==============================

系统要求：
  - Ubuntu 20.04 / 22.04（64位）
  - NVIDIA 显卡 + 已安装驱动（nvidia-smi 可用）
  - 至少 16 GB 磁盘空间（解压需要）

步骤：
  1. 解压本压缩包到任意目录
  2. 打开终端，进入解压目录
  3. 首次安装（只需一次）：
       ./install.sh
  4. 运行演示：
       ./run.sh

常用参数：
  ./run.sh --step          逐步模式，适合课堂讲解
  ./run.sh --loops 5       循环演示 5 次
  ./run.sh --speed 1.0     1 倍速（默认 2 倍速）
  ./run.sh --vis-grasp     显示点云可视化（会暂停）

遇到问题？
  - 黑屏/无窗口：确认 NVIDIA 驱动正常（nvidia-smi 有输出）
  - 段错误：尝试 export LIBGL_DEBUG=verbose 后再运行
  - 其他：联系任课老师
README_EOF

# ── 打包成最终 tar.gz ──────────────────────────────────────
echo "  正在打包所有文件（最终压缩，env 已压缩跳过二次压缩）..."
tar -czf "$OUTPUT_TAR" -C "$DEMO_DIR" student_package \
    --transform 's|^student_package|g1_grasp_demo_offline|'

# 清理临时目录
rm -rf "$OUTPUT_DIR"

# ── 完成 ───────────────────────────────────────────────────
FINAL_SIZE=$(du -sh "$OUTPUT_TAR" | cut -f1)
echo ""
echo "=================================================="
echo "  打包完成！"
echo "  输出文件: $OUTPUT_TAR"
echo "  文件大小: $FINAL_SIZE"
echo ""
echo "  分发给学生后，学生操作步骤："
echo "    tar -xzf g1_grasp_demo_offline.tar.gz"
echo "    cd g1_grasp_demo_offline"
echo "    ./install.sh"
echo "    ./run.sh"
echo "=================================================="
