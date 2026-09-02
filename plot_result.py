import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ==========================================
# ディレクトリ設定
# ==========================================
NAME = input("Enter your name: ")
ALGORITHM = input("Enter your algorithm: ")
SELECT = input("Select figure: ")

METHOD = None
TEMPERATURE = None

if SELECT == "alpha":
    TEMPERATURE = input("temperature: ")
else:
    METHOD = input("Select method: ")
    if METHOD == "method":
        TEMPERATURE = "*"

if ALGORITHM == "teian":
    exp_id = input("Enter experiment ID: ")
    ALPHA_MIN = float(input("Enter alpha min: "))
    ALPHA_MAX = float(input("Enter alpha max: "))

ROOT_DIR = Path("plot") / NAME

# ==============================
# 提案手法パラメータ
# ==============================

# ==============================
# グラフ設定
# ==============================
FIG_SIZE = (10, 6)

if NAME == "humanoid":
    MIN_REWARD = 0
    MAX_REWARD = 12000
elif NAME == "walker2d":
    MIN_REWARD = 0
    MAX_REWARD = 8000
elif NAME == "reacher":
    MIN_REWARD = -5
    MAX_REWARD = -2.9
elif NAME == "hopper":
    MIN_REWARD = 0
    MAX_REWARD = 4000    

WINDOW = 300
LINE_WIDTH = 5
STD_ALPHA = 0.2
TITLE_SIZE = 30
XLABEL_SIZE = 20
YLABEL_SIZE = 20
LEGEND_SIZE = 15
TICK_SIZE = 17
DPI = 300
SHOW_STD = True

SEED_COLUMNS = [
    "seed_2021",
    "seed_2022",
    "seed_2023",
    "seed_2024",
    "seed_2025",
]


def get_temp_dirs(method):
    """指定した手法の温度ディレクトリ一覧を返す。"""
    if method == "teian":
        return []

    search_method = "clr" if method == "method" else method
    pattern = ROOT_DIR / search_method / ALGORITHM / "*"
    print(sorted(
        os.path.basename(path)
        for path in glob.glob(str(pattern))
        if os.path.isdir(path)
    ))
    return sorted(
        os.path.basename(path)
        for path in glob.glob(str(pattern))
        if os.path.isdir(path)
    )


def preprocess(df):
    """各seedの平均・標準偏差・移動平均を作る。"""
    df["mean"] = df[SEED_COLUMNS].mean(axis=1)
    df["std"] = df[SEED_COLUMNS].std(axis=1)
    df["mean_smooth"] = df["mean"].rolling(WINDOW, min_periods=1).mean()
    df["std_smooth"] = df["std"].rolling(WINDOW, min_periods=1).mean()
    return df


def smooth_column(df, column):
    """指定列を移動平均で平滑化する。"""
    df[column] = df[column].rolling(WINDOW, min_periods=1).mean()
    return df


def style_axes(title, ylabel, ylim=None):
    """グラフ共通の見た目を設定する。"""
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Step", fontsize=XLABEL_SIZE)
    plt.ylabel(ylabel, fontsize=YLABEL_SIZE)
    plt.title(title, fontsize=TITLE_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.grid(True)


def plot_line(x, y, label):
    """線を描画し、その線に実際に割り当てられた色を返す。"""
    (line,) = plt.plot(x, y, label=label, linewidth=LINE_WIDTH)
    return line.get_color()


def fill_std_band(df, color):
    """平均±標準偏差の帯を、対応する線と同じ色で描画する。"""
    plt.fill_between(
        df["step"],
        df["mean_smooth"] - df["std_smooth"],
        df["mean_smooth"] + df["std_smooth"],
        alpha=STD_ALPHA,
        color=color,
    )


def read_my_metrics(method_name, temp):
    """my系手法の eval_metrics.csv を読み込んで前処理する。"""
    file_path = ROOT_DIR / method_name / ALGORITHM / temp / "eval_metrics.csv"
    if not file_path.exists():
        return None
    return preprocess(pd.read_csv(file_path))


def save_current_figure(file_name):
    plt.savefig(ROOT_DIR / file_name, dpi=DPI)


def plot_alpha_comparison():
    clr_dir = ROOT_DIR / "clr" / ALGORITHM / TEMPERATURE
    my_df = read_my_metrics("my", TEMPERATURE)
    my_action_df = read_my_metrics("my_action!=mean", TEMPERATURE)

    clr_alpha = smooth_column(pd.read_csv(clr_dir / "alpha.csv"), "Value")
    clr_reward = smooth_column(pd.read_csv(clr_dir / "reward.csv"), "Value")
    smooth_column(my_df, "alpha")
    smooth_column(my_df, "avg_reward")
    smooth_column(my_action_df, "alpha")
    smooth_column(my_action_df, "avg_reward")

    plt.figure(figsize=FIG_SIZE)
    plot_line(clr_alpha["Step"], clr_alpha["Value"], "CleanRL")
    plot_line(my_df["step"], my_df["alpha"], "My Method")
    plot_line(my_action_df["step"], my_action_df["alpha"], "My Method_action")
    style_axes("Alpha Transition", "Alpha", ylim=(0, 0.15))
    save_current_figure(f"alpha_comparison_{TEMPERATURE}.png")

    plt.figure(figsize=FIG_SIZE)
    plot_line(clr_reward["Step"], clr_reward["Value"], "CleanRL")
    my_color = plot_line(my_df["step"], my_df["avg_reward"], "My Method")
    my_action_color = plot_line(
        my_action_df["step"], my_action_df["avg_reward"], "My Method_action"
    )

    if SHOW_STD:
        fill_std_band(my_df, my_color)
        fill_std_band(my_action_df, my_action_color)

    style_axes("Reward Transition", "Reward")
    save_current_figure(f"reward_comparison_{TEMPERATURE}.png")


def plot_metric_for_temps(method_name, metric, title_name, ylabel, output_name, temp_dirs):
    plt.figure(figsize=FIG_SIZE)

    for temp in temp_dirs:
        df = read_my_metrics(method_name, temp)
        if df is None:
            continue

        if metric == "avg_reward" and method_name == "my":
            print(f"T={temp} Final Reward: {df['avg_reward'].tail(5).mean()}")

        smooth_column(df, metric)
        color = plot_line(df["step"], df[metric], f"T={temp}")

        if SHOW_STD and metric == "avg_reward":
            fill_std_band(df, color)

    ylim = (MIN_REWARD, MAX_REWARD) if metric == "avg_reward" else None
    style_axes(f"{title_name} {ylabel} Comparison ({ALGORITHM})", ylabel, ylim=ylim)
    save_current_figure(output_name)
    plt.show()


def plot_cleanrl_rewards(temp_dirs):
    plt.figure(figsize=FIG_SIZE)

    for temp in temp_dirs:
        reward_file = ROOT_DIR / "clr" / ALGORITHM / temp / "reward.csv"
        if not reward_file.exists():
            continue

        df = smooth_column(pd.read_csv(reward_file), "Value")
        plot_line(df["Step"], df["Value"], f"T={temp}")

    style_axes(
        f"CleanRL Reward Comparison ({ALGORITHM})",
        "Reward",
        ylim=(MIN_REWARD, MAX_REWARD),
    )
    save_current_figure("clr_reward_comparison.png")
    plt.show()


def get_teian_dirs():
    """
    提案手法の実験ディレクトリを取得する。

    基本となる v1 と、
    exp_id 配下にある tau= で始まるディレクトリを取得する。
    """
    base_dir = (
        ROOT_DIR
        / "my"
        / ALGORITHM
        / f"[{ALPHA_MIN}, {ALPHA_MAX}]"
        / f"{exp_id}"
    )

    dirs = []

    # 従来の v1
    v1_dir = base_dir / "v1"
    if v1_dir.is_dir():
        dirs.append(("v1", v1_dir))

    # tau= で始まるディレクトリ
    tau_dirs = sorted(
        [
            path
            for path in base_dir.iterdir()
            if path.is_dir() and path.name.startswith("tau=")
        ],
        key=lambda x: x.name
    )

    for tau_dir in tau_dirs:
        # tau=0.01/ex2=3000/v1 のような構造
        ex2_dirs = [
            path
            for path in tau_dir.iterdir()
            if path.is_dir() and path.name.startswith("ex2=")
        ]

        for ex2_dir in sorted(ex2_dirs, key=lambda x: x.name):
            v1_dir = ex2_dir / "v1"

            if v1_dir.is_dir():
                label = f"{tau_dir.name}/{ex2_dir.name}"
                dirs.append((label, v1_dir))

    return dirs


def read_teian_metrics(base_dir, index):
    """
    指定された実験ディレクトリから
    eval_metrics{index}.csv を読み込む。
    """
    file_path = (
        base_dir
        / "2021"
        / "True"
        / f"eval_metrics{index}.csv"
    )

    if not file_path.exists():
        return None, file_path

    return preprocess(pd.read_csv(file_path)), file_path

def plot_teian_metric(metric, ylabel, output_name):
    plt.figure(figsize=FIG_SIZE)

    teian_dirs = get_teian_dirs()

    if not teian_dirs:
        print("No Teian experiment directories found.")
        return

    for base_label, base_dir in teian_dirs:

        for index, alpha in enumerate([ALPHA_MIN, ALPHA_MAX]):

            df, file_path = read_teian_metrics(base_dir, index)

            if df is None:
                print(f"File not found: {file_path}")
                continue

            smooth_column(df, metric)

            # v1 は T=alpha、
            # tau=... は tau/ex2 の情報も含める
            if base_label == "v1":
                label = f"T={alpha}"
            else:
                label = f"{base_label}, T={alpha}"

            color = plot_line(
                df["step"],
                df[metric],
                label
            )

            if metric != "avg_reward":
                final_reward = df["avg_reward"].tail(5).mean()

                print(
                    f"{base_label}, T={alpha} "
                    f"Final Reward: {final_reward}"
                )

                with open(
                    file_path.parent / "final_rewards.csv",
                    "a"
                ) as f:
                    f.write(
                        f"{base_label},{alpha},{final_reward}\n"
                    )

            if SHOW_STD and metric == "avg_reward":
                fill_std_band(df, color)

    ylim = (
        (MIN_REWARD, MAX_REWARD)
        if metric == "avg_reward"
        else None
    )

    style_axes(
        f"Teian {ylabel} Comparison ({ALGORITHM})",
        ylabel,
        ylim=ylim
    )

    save_current_figure(output_name)
    plt.savefig(file_path.parent /output_name, dpi=DPI)

if SELECT == "alpha":
    plot_alpha_comparison()
else:
    TEMP_DIRS = get_temp_dirs(METHOD)

    if METHOD in ("clr", "method"):
        plot_cleanrl_rewards(TEMP_DIRS)

    if METHOD in ("my", "method"):
        plot_metric_for_temps(
            "my", "avg_reward", "My Method", "Reward", "my_reward_comparison.png", TEMP_DIRS
        )
        plot_metric_for_temps(
            "my",
            "policy_loss",
            "My Method",
            "Policy Loss",
            "my_policy_loss_comparison.png",
            TEMP_DIRS,
        )
        plot_metric_for_temps(
            "my",
            "critic1_loss",
            "My Method",
            "Critic1 Loss",
            "my_critic1_loss_comparison.png",
            TEMP_DIRS,
        )

    if METHOD in ("teian", "method"):
        plot_teian_metric("avg_reward", "Reward", "teian_reward_comparison.png")
        plot_teian_metric("policy_loss", "Policy Loss", "teian_policy_loss_comparison.png")
        plot_teian_metric("critic1_loss", "Critic Loss", "teian_critic_loss_comparison.png")

    if METHOD in ("my_action", "method"):
        plot_metric_for_temps(
            "my_action!=mean",
            "avg_reward",
            "My_action",
            "Reward",
            "my_action_reward_comparison.png",
            TEMP_DIRS,
        )
