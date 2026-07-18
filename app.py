"""B205梁结构健康智能推演系统 V1。

健康指数由裂缝宽度、应变和挠度计算，维修方案通过改变退化速度进行推演。
"""

from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="B205梁结构健康智能推演系统",
    page_icon="🏗️",
    layout="wide",
)

# 绘图函数共用的中文字体对象；启动后由 set_chinese_font() 更新
CHINESE_FONT_PROPERTIES = font_manager.FontProperties()


def set_chinese_font():
    """优先加载项目内中文字体，缺失时安全回退到系统字体。"""
    global CHINESE_FONT_PROPERTIES

    # 使用 __file__ 构建绝对路径，兼容本地 Windows 和 Streamlit Cloud
    project_font_path = (
        Path(__file__).resolve().parent / "fonts" / "NotoSansSC-Regular.ttf"
    )

    if project_font_path.exists():
        try:
            font_manager.fontManager.addfont(str(project_font_path))
            CHINESE_FONT_PROPERTIES = font_manager.FontProperties(
                fname=str(project_font_path)
            )
            font_name = CHINESE_FONT_PROPERTIES.get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            return CHINESE_FONT_PROPERTIES
        except (OSError, RuntimeError, ValueError, AttributeError):
            # 字体文件不可读时继续执行系统字体回退，避免网页崩溃
            pass

    # 安全回退：选择当前系统中真实存在的中文字体文件
    preferred_fonts = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "Source Han Sans CN",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
    ]
    system_font = next(
        (
            font
            for preferred_name in preferred_fonts
            for font in font_manager.fontManager.ttflist
            if font.name == preferred_name
        ),
        None,
    )

    if system_font is not None:
        CHINESE_FONT_PROPERTIES = font_manager.FontProperties(fname=system_font.fname)
        plt.rcParams["font.family"] = system_font.name
        plt.rcParams["font.sans-serif"] = [system_font.name, "DejaVu Sans"]
    else:
        CHINESE_FONT_PROPERTIES = font_manager.FontProperties()

    plt.rcParams["axes.unicode_minus"] = False
    return CHINESE_FONT_PROPERTIES


@st.cache_data
def generate_monitoring_data(days=30):
    """生成最近若干天的模拟监测数据。"""
    # 固定随机种子，使每次刷新页面时数据保持一致
    rng = np.random.default_rng(205)
    times = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
    day_number = np.arange(days)

    # 三项指标随时间缓慢增加，并加入少量随机波动
    crack_width = 0.18 + day_number * 0.002 + rng.normal(0, 0.008, days)
    strain = 410 + day_number * 1.8 + rng.normal(0, 9, days)
    deflection = 4.20 + day_number * 0.025 + rng.normal(0, 0.10, days)

    return pd.DataFrame(
        {
            "时间": times,
            "裂缝宽度（mm）": np.round(crack_width, 3),
            "应变（με）": np.round(strain, 1),
            "挠度（mm）": np.round(deflection, 2),
        }
    )


def calculate_health_index(crack_width, strain, deflection):
    """根据裂缝宽度、应变和挠度计算0～100分健康指数。

    这是便于验证流程的简化模型：先计算各指标相对参考值的程度，
    再按照裂缝40%、应变35%、挠度25%的权重计算综合损伤。
    """
    reference_values = {
        "crack_width": 0.40,  # 裂缝宽度参考值（mm）
        "strain": 800.0,  # 应变参考值（με）
        "deflection": 10.0,  # 挠度参考值（mm）
    }

    crack_ratio = max(crack_width, 0) / reference_values["crack_width"]
    strain_ratio = max(strain, 0) / reference_values["strain"]
    deflection_ratio = max(deflection, 0) / reference_values["deflection"]

    damage_score = (
        0.40 * crack_ratio
        + 0.35 * strain_ratio
        + 0.25 * deflection_ratio
    )
    health_index = 100 - 25 * damage_score
    return float(np.clip(health_index, 0, 100))


def get_current_health(monitoring_data):
    """使用最新一条监测记录计算当前健康指数。"""
    latest = monitoring_data.iloc[-1]
    return calculate_health_index(
        latest["裂缝宽度（mm）"],
        latest["应变（με）"],
        latest["挠度（mm）"],
    )


def get_risk_level(health_index):
    """根据健康指数给出简单风险等级。"""
    if health_index >= 90:
        return "🟢 绿色正常"
    if health_index >= 80:
        return "🟡 黄色关注"
    if health_index >= 60:
        return "🟠 橙色预警"
    return "🔴 红色警报"


def calculate_degradation_rate(monitoring_data):
    """根据最新监测指标的严重程度估计基础日退化速度。"""
    latest = monitoring_data.iloc[-1]
    severity = (
        0.40 * latest["裂缝宽度（mm）"] / 0.40
        + 0.35 * latest["应变（με）"] / 800.0
        + 0.25 * latest["挠度（mm）"] / 10.0
    )
    # 当前模拟数据下，基础退化速度约为每天0.05分
    return 0.025 + 0.04 * severity


def predict_health_trend(monitoring_data, days=180, current_health=None):
    """根据当前健康指数和监测数据预测无维修健康趋势。"""
    # 未传入模拟健康指数时，仍按原方式从监测数据计算
    if current_health is None:
        current_health = get_current_health(monitoring_data)
    daily_rate = calculate_degradation_rate(monitoring_data)
    future_days = np.arange(days + 1)
    future_dates = [date.today() + timedelta(days=int(day)) for day in future_days]

    # 随时间增加轻微加速项，模拟结构自然老化
    health = current_health - daily_rate * future_days - 0.00005 * future_days**2
    return future_dates, np.clip(health, 0, 100)


def simulate_counterfactuals(monitoring_data, days=180, current_health=None):
    """通过改变退化速度，模拟三种维修方案的未来结果。"""
    # 支持使用滑动条调整后得到的新健康状态
    if current_health is None:
        current_health = get_current_health(monitoring_data)
    base_rate = calculate_degradation_rate(monitoring_data)
    future_days = np.arange(days + 1)
    future_dates = [date.today() + timedelta(days=int(day)) for day in future_days]

    # 不维修：保持原退化速度；小修和大修分别降低退化速度
    scenario_settings = {
        "情景A：不维修": {"rate_factor": 1.00, "acceleration": 0.00005},
        "情景B：小修方案": {"rate_factor": 0.45, "acceleration": 0.00002},
        "情景C：大修方案": {"rate_factor": 0.20, "acceleration": 0.000008},
    }

    scenarios = {}
    for name, setting in scenario_settings.items():
        health = (
            current_health
            - base_rate * setting["rate_factor"] * future_days
            - setting["acceleration"] * future_days**2
        )
        scenarios[name] = np.clip(health, 0, 100)

    return future_dates, scenarios


def draw_health_chart(dates, health):
    """绘制未来健康趋势图。"""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, health, color="#E67E22", linewidth=2.5, label="预测健康指数")

    # 在曲线起点突出显示滑动条对应的当前健康状态
    current_health = float(health[0])
    ax.scatter(
        dates[0],
        current_health,
        color="#C0392B",
        s=90,
        edgecolors="white",
        linewidth=1.5,
        zorder=5,
        label="当前状态",
    )
    ax.annotate(
        f"当前健康指数 {current_health:.1f}分",
        xy=(dates[0], current_health),
        xytext=(18, 18),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        color="#C0392B",
        fontproperties=CHINESE_FONT_PROPERTIES,
        arrowprops={"arrowstyle": "->", "color": "#C0392B"},
    )

    ax.axhline(80, color="#F1C40F", linestyle="--", alpha=0.8, label="关注线（80分）")
    ax.set_title(
        f"未来180天健康趋势预测（当前健康指数：{current_health:.1f}分）",
        fontproperties=CHINESE_FONT_PROPERTIES,
    )
    ax.set_xlabel("时间", fontproperties=CHINESE_FONT_PROPERTIES)
    ax.set_ylabel("健康指数", fontproperties=CHINESE_FONT_PROPERTIES)
    # 固定坐标范围，确保不同模拟状态的曲线高低位置可以直接比较
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(prop=CHINESE_FONT_PROPERTIES)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def draw_counterfactual_chart(dates, scenarios):
    """将三种维修情景绘制在同一张图中。"""
    colors = ["#E74C3C", "#F39C12", "#27AE60"]
    fig, ax = plt.subplots(figsize=(10, 5))

    for (name, health), color in zip(scenarios.items(), colors):
        ax.plot(dates, health, linewidth=2.5, label=name, color=color)

    current_health = float(next(iter(scenarios.values()))[0])
    ax.set_title(
        f"未来180天反事实推演（当前健康指数：{current_health:.1f}分）",
        fontproperties=CHINESE_FONT_PROPERTIES,
    )
    ax.set_xlabel("时间", fontproperties=CHINESE_FONT_PROPERTIES)
    ax.set_ylabel("健康指数", fontproperties=CHINESE_FONT_PROPERTIES)
    # 与健康趋势图使用相同的固定范围，避免自动缩放造成视觉误差
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(prop=CHINESE_FONT_PROPERTIES)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def draw_digital_twin_section():
    """绘制教学楼二层B205梁的轻量化数字孪生定位剖面。"""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    # 建筑剖面主体：一层、二层、三层按真实空间关系自下而上排列
    building_left = 2.0
    building_width = 7.2
    floor_bottoms = [1.1, 3.0, 4.9]
    floor_labels = ["一层", "二层", "三层"]

    for floor_bottom, floor_label in zip(floor_bottoms, floor_labels):
        ax.add_patch(
            Rectangle(
                (building_left, floor_bottom),
                building_width,
                1.7,
                facecolor="#F4F7FA",
                edgecolor="#D5E0E8",
                linewidth=1.0,
            )
        )
        ax.text(
            1.45,
            floor_bottom + 0.85,
            floor_label,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#314E63",
            fontproperties=CHINESE_FONT_PROPERTIES,
        )

    # 楼板和屋面
    for slab_y in [1.0, 2.9, 4.8, 6.7]:
        ax.add_patch(
            Rectangle(
                (building_left - 0.15, slab_y),
                building_width + 0.3,
                0.13,
                facecolor="#71899B",
                edgecolor="none",
            )
        )

    # 规则柱网，体现建筑空间和结构关系
    column_x_positions = [2.05, 4.4, 6.75, 9.05]
    for column_x in column_x_positions:
        ax.add_patch(
            Rectangle(
                (column_x, 1.1),
                0.13,
                5.6,
                facecolor="#AABBC7",
                edgecolor="none",
            )
        )

    # 周边普通梁采用浅灰色
    for beam_y in [2.65, 4.55, 6.45]:
        ax.add_patch(
            Rectangle(
                (2.18, beam_y),
                6.87,
                0.16,
                facecolor="#C4D0D9",
                edgecolor="#9FB1BE",
                linewidth=0.7,
            )
        )

    # B205位于二层梁结构位置，黄色高亮并增加选中边界
    ax.add_patch(
        Rectangle(
            (4.4, 3.15),
            2.35,
            0.55,
            facecolor="none",
            edgecolor="#D39A00",
            linewidth=1.5,
            linestyle="--",
        )
    )
    ax.add_patch(
        Rectangle(
            (4.52, 3.31),
            2.1,
            0.23,
            facecolor="#F4B400",
            edgecolor="#806000",
            linewidth=1.2,
        )
    )
    ax.scatter(
        [4.7, 6.45],
        [3.425, 3.425],
        s=18,
        color="#FFF1A8",
        edgecolors="#806000",
        linewidths=0.8,
        zorder=5,
    )
    ax.text(
        5.57,
        3.82,
        "B205梁",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="#624A00",
        fontproperties=CHINESE_FONT_PROPERTIES,
    )

    # 定位引线和当前风险状态
    ax.annotate(
        "构件定位：教学楼二层\n当前状态：黄色关注",
        xy=(6.62, 3.43),
        xytext=(9.75, 4.1),
        ha="left",
        va="center",
        fontsize=10.5,
        color="#3E5261",
        fontproperties=CHINESE_FONT_PROPERTIES,
        arrowprops={"arrowstyle": "->", "color": "#D39A00", "linewidth": 1.6},
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#FFF8DD", "edgecolor": "#D8B74C"},
    )

    # 状态图例
    legend_items = [
        ("#2E7D32", "绿色：安全区域"),
        ("#F4B400", "黄色：关注区域"),
        ("#C62828", "红色：风险区域"),
    ]
    legend_x = [2.1, 5.0, 7.9]
    for x_position, (color, label) in zip(legend_x, legend_items):
        ax.scatter(x_position, 0.42, s=55, color=color, zorder=4)
        ax.text(
            x_position + 0.2,
            0.42,
            label,
            ha="left",
            va="center",
            fontsize=9.5,
            color="#405566",
            fontproperties=CHINESE_FONT_PROPERTIES,
        )

    ax.set_title(
        "教学楼二层B205梁数字孪生对象定位",
        fontsize=14,
        fontweight="bold",
        color="#173B5E",
        pad=10,
        fontproperties=CHINESE_FONT_PROPERTIES,
    )
    fig.tight_layout(pad=0.8)
    return fig


def generate_ai_recommendation(scenarios):
    """根据各方案最终健康指数、成本和施工影响自动生成建议。"""
    no_repair_final = float(scenarios["情景A：不维修"][-1])
    minor_repair_final = float(scenarios["情景B：小修方案"][-1])
    major_repair_final = float(scenarios["情景C：大修方案"][-1])

    minor_improvement = minor_repair_final - no_repair_final
    major_extra_improvement = major_repair_final - minor_repair_final

    # 简单决策规则：先满足安全要求，再比较额外收益、成本和施工影响
    if no_repair_final >= 85 and minor_improvement < 2:
        recommendation = (
            f"根据反事实推演结果，不维修情况下180天后健康指数仍为"
            f"{no_repair_final:.1f}分；小修仅额外提高{minor_improvement:.1f}分。"
            "不维修方案无需额外维修费用，可采用人工巡检、裂缝记录和状态跟踪。"
            "当前可暂不维修，但应继续加强监测，并根据指标变化及时复核。"
        )
    elif minor_repair_final >= 80 and major_extra_improvement < 5:
        recommendation = (
            f"根据反事实推演结果，不维修情况下180天后健康指数下降至"
            f"{no_repair_final:.1f}分；小修方案通过环氧树脂灌浆、碳纤维布局部增强"
            f"和表面防护处理，可使健康指数提升至{minor_repair_final:.1f}分，"
            f"相比不维修提高{minor_improvement:.1f}分。大修方案虽可再提高"
            f"{major_extra_improvement:.1f}分，但整体加固成本约15000～30000元/根梁，"
            "施工影响更大；小修成本约3000～8000元/根梁。"
            "综合安全提升、成本和施工周期，推荐采用小修方案。"
        )
    else:
        recommendation = (
            f"根据反事实推演结果，不维修、小修和大修方案180天后的健康指数分别为"
            f"{no_repair_final:.1f}分、{minor_repair_final:.1f}分和"
            f"{major_repair_final:.1f}分。小修相比不维修提高{minor_improvement:.1f}分，"
            f"大修相比小修再提高{major_extra_improvement:.1f}分。"
            "由于小修后的健康裕度仍然不足，建议采用外包钢、增大截面和节点强化等"
            "整体加固措施，预计成本15000～30000元/根梁。推荐采用大修方案，"
            "并由专业工程师复核具体加固设计。"
        )

    return recommendation


def main():
    """页面主程序。"""
    set_chinese_font()

    # 轻量级蓝白工业风样式，仅影响页面展示
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f5f8fc;
            font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
            color: #24384a;
            color-scheme: light;
        }
        /* 全局正文与组件文字固定为深色，避免云端主题产生白字 */
        .stApp p,
        .stApp li,
        .stApp label,
        .stApp small,
        .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stWidgetLabel"],
        .stApp [data-testid="stExpander"] summary,
        .stApp [data-testid="stAlert"] {
            color: #2f3e4c !important;
        }
        .stApp [data-testid="stExpander"] summary p,
        .stApp [data-testid="stAlert"] p,
        .stApp [data-testid="stAlert"] li,
        .stApp [data-testid="stMetricDelta"] {
            color: #40576a !important;
        }
        .stApp input,
        .stApp textarea,
        .stApp select {
            color: #24384a !important;
            background-color: #ffffff !important;
        }
        /* 页面主标题：突出平台名称 */
        h1 {
            color: #123b60 !important;
            font-size: 2.65rem !important;
            font-weight: 800 !important;
            line-height: 1.2 !important;
            letter-spacing: 0.01em;
            margin-bottom: 0.25rem !important;
        }
        /* 一级模块标题：形成清晰的业务区块层级 */
        h2 {
            color: #173b5e !important;
            font-size: 1.65rem !important;
            font-weight: 750 !important;
            line-height: 1.35 !important;
            padding-bottom: 0.35rem;
            border-bottom: 2px solid #dce8f3;
        }
        h3 {
            color: #24577f !important;
            font-size: 1.22rem !important;
            font-weight: 700 !important;
        }
        /* 副标题和辅助说明弱化显示 */
        [data-testid="stCaptionContainer"] p {
            color: #6f7f8f !important;
            font-size: 0.9rem !important;
            line-height: 1.55 !important;
        }
        .stMarkdown p, .stMarkdown li {
            line-height: 1.65;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d7e3f1;
            border-left: 4px solid #1769aa;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 2px 8px rgba(31, 78, 121, 0.06);
        }
        /* 核心指标数字加大，增强数字孪生仪表盘感 */
        [data-testid="stMetricValue"] {
            color: #123f66;
            font-size: 1.8rem !important;
            font-weight: 800 !important;
            line-height: 1.25 !important;
        }
        [data-testid="stMetricLabel"] {
            color: #5d7184;
            font-size: 0.88rem !important;
            font-weight: 600 !important;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #d7e3f1;
            border-radius: 8px;
        }
        .platform-flow {
            padding: 14px;
            margin: 8px 0 22px 0;
            text-align: center;
            color: #174f7c;
            background: #eaf3fb;
            border: 1px solid #c9deef;
            border-radius: 8px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🏗️ B205梁结构健康智能推演系统")
    st.caption("基于AI预测与反事实推演的结构维修决策验证原型")
    st.info("当前为 V1 模拟验证版本，未调用真实AI模型。")

    st.subheader("系统概览")
    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
    overview_col1.metric("平台", "数字孪生智能决策")
    overview_col2.metric("分析对象", "B-2-05 混凝土梁")
    overview_col3.metric("应用场景", "教学楼健康监测")
    overview_status_placeholder = overview_col4.empty()

    # 数字孪生对象展示位置
    st.markdown("### B205数字孪生对象定位")
    twin_col1, twin_col2 = st.columns([2, 1])
    twin_diagram_placeholder = twin_col1.empty()
    twin_info_placeholder = twin_col2.empty()

    st.markdown("**系统工作流程**")
    st.markdown(
        """
        <div class="platform-flow">
        传感数据　→　工程指标计算　→　健康状态评价　→　AI风险分析　→　未来趋势预测　→　反事实推演　→　维修决策
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.header("1. 🏗️ 构件基础信息")
    col1, col2, col3 = st.columns(3)
    col1.metric("构件编号", "B-2-05")
    col2.metric("类型", "钢筋混凝土梁")
    col3.metric("位置", "教学楼二层")

    # 先生成监测数据，再由最新一条数据计算当前健康指数
    monitoring_data = generate_monitoring_data()
    current_health = get_current_health(monitoring_data)
    risk_level = get_risk_level(current_health)

    st.header("2. 🧭 当前结构状态")
    status_col1, status_col2, status_col3 = st.columns(3)
    # 先预留显示位置，在滑动条计算完成后填入动态结果
    health_status_placeholder = status_col1.empty()
    risk_status_placeholder = status_col2.empty()
    status_col3.metric("预测周期", "180天")

    st.header("3. 📡 结构监测数据")
    st.dataframe(
        monitoring_data.sort_values("时间", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("说明：以上为最近30天的模拟监测数据，仅用于验证原型流程。")

    st.header("4. 🎛️ 结构状态情景模拟")
    st.write(
        "通过调整结构响应参数，模拟不同损伤状态下建筑构件未来演化过程，"
        "用于验证反事实推演和智能决策能力。"
    )

    # 使用最新监测值作为三个滑动条的默认值
    latest_data = monitoring_data.iloc[-1]
    slider_col1, slider_col2, slider_col3 = st.columns(3)
    simulated_crack = slider_col1.slider(
        "裂缝宽度（mm）",
        min_value=0.10,
        max_value=0.50,
        value=float(np.clip(round(latest_data["裂缝宽度（mm）"], 2), 0.10, 0.50)),
        step=0.01,
    )
    simulated_strain = slider_col2.slider(
        "应变（με）",
        min_value=300,
        max_value=800,
        value=int(np.clip(round(latest_data["应变（με）"] / 10) * 10, 300, 800)),
        step=10,
    )
    simulated_deflection = slider_col3.slider(
        "挠度（mm）",
        min_value=2.0,
        max_value=10.0,
        value=float(np.clip(round(latest_data["挠度（mm）"], 1), 2.0, 10.0)),
        step=0.1,
    )

    # 调用原有公式，使用滑动条输入重新计算健康指数
    current_health = calculate_health_index(
        simulated_crack,
        simulated_strain,
        simulated_deflection,
    )
    risk_level = get_risk_level(current_health)

    # 将调整值写入监测数据副本，原始模拟监测表保持不变
    simulated_data = monitoring_data.copy()
    latest_index = simulated_data.index[-1]
    simulated_data.at[latest_index, "裂缝宽度（mm）"] = simulated_crack
    simulated_data.at[latest_index, "应变（με）"] = simulated_strain
    simulated_data.at[latest_index, "挠度（mm）"] = simulated_deflection

    st.markdown("**模拟状态更新结果：**")
    result_col1, result_col2 = st.columns(2)
    result_col1.metric("健康指数", f"{current_health:.1f}%")
    result_col2.metric("风险等级", risk_level)

    # 同步更新页面顶部的“当前结构状态”
    health_status_placeholder.metric("健康指数", f"{current_health:.1f} / 100")
    risk_status_placeholder.metric("风险等级", risk_level)
    overview_status_placeholder.metric(
        "当前状态",
        f"{current_health:.1f}%",
        risk_level,
    )

    # 当前数字孪生基准状态：B205梁为黄色关注构件
    twin_diagram_placeholder.markdown(
        """
        <div style="background:#ffffff;border:1px solid #d7e3f1;border-radius:8px;
                    padding:12px;box-shadow:0 2px 8px rgba(31,78,121,0.06);">
        <svg viewBox="0 0 760 410" width="100%" role="img"
             aria-label="教学楼二层B205梁数字孪生对象定位视图"
             style="font-family:'Microsoft YaHei','Segoe UI',Arial,sans-serif;">
          <!-- 技术视图标题栏 -->
          <rect x="12" y="12" width="736" height="43" rx="5"
                fill="#eaf3fb" stroke="#bfd5e6"/>
          <text x="30" y="40" font-size="19" font-weight="700" fill="#173b5e">
            教学楼二层 B205梁｜数字孪生对象定位
          </text>
          <text x="630" y="39" font-size="13" fill="#60788b">SECTION VIEW</text>

          <!-- 教学楼三层剖面：楼层从下向上依次为一层、二层、三层 -->
          <rect x="100" y="78" width="445" height="252" fill="#f8fbfd"
                stroke="#315f83" stroke-width="3"/>
          <rect x="103" y="81" width="439" height="80" fill="#f1f6f9"/>
          <rect x="103" y="164" width="439" height="80" fill="#f9fbfc"/>
          <rect x="103" y="247" width="439" height="80" fill="#f1f6f9"/>

          <!-- 楼板和屋面 -->
          <rect x="91" y="75" width="463" height="8" fill="#6f899d"/>
          <rect x="91" y="158" width="463" height="8" fill="#8ea5b6"/>
          <rect x="91" y="241" width="463" height="8" fill="#8ea5b6"/>
          <rect x="85" y="327" width="475" height="10" fill="#58758c"/>

          <!-- 规则柱网，表达真实构件空间关系 -->
          <rect x="101" y="80" width="10" height="249" fill="#a9bac7"/>
          <rect x="210" y="80" width="10" height="249" fill="#a9bac7"/>
          <rect x="322" y="80" width="10" height="249" fill="#a9bac7"/>
          <rect x="434" y="80" width="10" height="249" fill="#a9bac7"/>
          <rect x="533" y="80" width="10" height="249" fill="#a9bac7"/>

          <!-- 各楼层空间标注，按真实高程从上到下排列 -->
          <text x="27" y="124" font-size="18" font-weight="700" fill="#405b70">三层</text>
          <text x="27" y="207" font-size="18" font-weight="700" fill="#173b5e">二层</text>
          <text x="27" y="290" font-size="18" font-weight="700" fill="#405b70">一层</text>
          <line x1="67" y1="118" x2="90" y2="118" stroke="#9aafbf" stroke-width="2"/>
          <line x1="67" y1="201" x2="90" y2="201" stroke="#315f83" stroke-width="2"/>
          <line x1="67" y1="284" x2="90" y2="284" stroke="#9aafbf" stroke-width="2"/>

          <!-- 普通梁构件保持浅色 -->
          <rect x="111" y="143" width="99" height="11" fill="#c3d0d9"/>
          <rect x="220" y="143" width="102" height="11" fill="#c3d0d9"/>
          <rect x="332" y="143" width="102" height="11" fill="#c3d0d9"/>
          <rect x="444" y="143" width="89" height="11" fill="#c3d0d9"/>
          <rect x="111" y="226" width="99" height="11" fill="#c3d0d9"/>
          <rect x="444" y="226" width="89" height="11" fill="#c3d0d9"/>
          <rect x="111" y="310" width="422" height="11" fill="#c3d0d9"/>

          <!-- B205梁：位于二层柱网中，黄色高亮并显示选中边界 -->
          <rect x="204" y="207" width="246" height="42" rx="4"
                fill="none" stroke="#d89c00" stroke-width="2" stroke-dasharray="7 5"/>
          <rect x="220" y="222" width="214" height="17" rx="3"
                fill="#F4B400" stroke="#8b6800" stroke-width="2"/>
          <text x="295" y="216" font-size="17" font-weight="800" fill="#5f4900">B205</text>
          <circle cx="240" cy="230" r="4" fill="#fff4bf" stroke="#806000"/>
          <circle cx="414" cy="230" r="4" fill="#fff4bf" stroke="#806000"/>

          <!-- 构件定位引线与状态面板 -->
          <polyline points="434,230 580,230 600,206" fill="none"
                    stroke="#d89c00" stroke-width="3"/>
          <circle cx="434" cy="230" r="6" fill="#F4B400" stroke="#806000"/>
          <rect x="580" y="112" width="160" height="94" rx="6"
                fill="#fff9e5" stroke="#d7b84a" stroke-width="2"/>
          <text x="598" y="139" font-size="14" fill="#6a6041">SELECTED ELEMENT</text>
          <text x="598" y="166" font-size="22" font-weight="800" fill="#173b5e">B205梁</text>
          <circle cx="603" cy="187" r="6" fill="#F4B400"/>
          <text x="617" y="193" font-size="16" font-weight="700" fill="#6b5300">黄色关注</text>

          <!-- 风险状态图例 -->
          <rect x="91" y="358" width="649" height="35" rx="5"
                fill="#f7fafc" stroke="#d5e1ea"/>
          <text x="108" y="381" font-size="14" font-weight="700" fill="#526a7c">状态图例</text>
          <circle cx="205" cy="376" r="7" fill="#2E7D32"/>
          <text x="220" y="382" font-size="14" fill="#405566">绿色：安全区域</text>
          <circle cx="365" cy="376" r="7" fill="#F4B400"/>
          <text x="380" y="382" font-size="14" fill="#405566">黄色：关注区域</text>
          <circle cx="535" cy="376" r="7" fill="#C62828"/>
          <text x="550" y="382" font-size="14" fill="#405566">红色：风险区域</text>
        </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )
    twin_info_placeholder.info(
        """
**数字孪生构件信息**

**构件编号：** B-2-05  
**类型：** 钢筋混凝土梁  
**位置：** 教学楼二层  
**状态：** 黄色关注  
**健康指数：** 85%

该视图将AI分析结果定位到教学楼二层的具体梁构件。
        """
    )

    # 使用 Matplotlib 可靠渲染建筑剖面，替换云端无法显示的内嵌SVG
    twin_figure = draw_digital_twin_section()
    twin_diagram_placeholder.pyplot(twin_figure, use_container_width=True)
    plt.close(twin_figure)

    st.header("5. 📈 健康趋势预测")
    future_dates, health = predict_health_trend(
        simulated_data,
        current_health=current_health,
    )
    trend_figure = draw_health_chart(future_dates, health)
    st.pyplot(trend_figure, use_container_width=True)
    plt.close(trend_figure)
    st.write(f"预计180天后健康指数约为 **{health[-1]:.1f}分**。")

    st.header("6. 🔮 反事实推演")
    st.write("对比不同维修方案通过改变结构退化速度，对未来180天健康状态产生的影响。")
    scenario_dates, scenarios = simulate_counterfactuals(
        simulated_data,
        current_health=current_health,
    )
    scenario_figure = draw_counterfactual_chart(scenario_dates, scenarios)
    st.pyplot(scenario_figure, use_container_width=True)
    plt.close(scenario_figure)

    # 三种方案概览卡片，帮助快速比较工程含义
    plan_col1, plan_col2, plan_col3 = st.columns(3)
    plan_col1.info(
        "**方案A｜不维修**\n\n保持当前退化趋势  \n风险持续增加"
    )
    plan_col2.success(
        "**方案B｜小修方案**\n\n裂缝修补 + 局部加固  \n成本适中，降低退化速度"
    )
    plan_col3.warning(
        "**方案C｜大修方案**\n\n整体加固  \n安全提升最大，但成本较高"
    )

    st.subheader("反事实推演假设")
    st.markdown(
        """
**情景A：不维修**  
保持当前结构损伤发展趋势，不进行人为干预。模型认为裂缝、应变和挠度指标持续增长，因此健康指数持续下降。

**情景B：小修方案**  
通过裂缝修补和局部加固降低损伤发展速度。模型中体现为降低未来健康指数衰减速率。

**情景C：大修方案**  
通过整体加固恢复结构性能。模型中体现为显著降低长期退化速度，并提高未来健康裕度。
        """
    )

    st.subheader("维修方案详情")
    st.markdown(
        """
**情景A：不维修**

- **维护策略：** 常规巡检和状态监测
- **措施：** 人工巡检、裂缝记录、状态跟踪
- **预计成本：** 0元（不含日常巡检）

---

**情景B：小修方案——裂缝修补 + 局部加固**

- **措施：** 环氧树脂灌浆修补裂缝、碳纤维布局部增强、表面防护处理
- **施工周期：** 3～7天
- **预计成本：** 3000～8000元/根梁
- **效果：** 降低裂缝扩展速度，提高局部承载能力。

---

**情景C：大修方案——整体加固**

- **措施：** 外包钢加固、增大截面加固、节点区域强化
- **施工周期：** 15～30天
- **预计成本：** 15000～30000元/根梁
- **效果：** 最大程度恢复结构安全储备。
        """
    )

    st.subheader("方案综合评价")
    evaluation_data = pd.DataFrame(
        {
            "方案": list(scenarios.keys()),
            "维修措施": [
                "人工巡检、裂缝记录、状态跟踪",
                "裂缝灌浆、碳纤维布局部增强、表面防护",
                "外包钢、增大截面、节点区域强化",
            ],
            "180天健康指数": [f"{values[-1]:.1f}分" for values in scenarios.values()],
            "预计成本": [
                "0元（不含日常巡检）",
                "3000～8000元/根梁",
                "15000～30000元/根梁",
            ],
            "施工周期": [
                "无维修施工",
                "3～7天",
                "15～30天",
            ],
            "综合评价": [
                "短期经济，但长期风险较高",
                "降低损伤速度，安全性与经济性较均衡",
                "安全储备最高，但成本和施工影响最大",
            ],
        }
    )
    st.dataframe(evaluation_data, use_container_width=True, hide_index=True)

    # AI风险解释与证据链仅用于展示，不参与已有模型计算
    with st.expander("🤖 AI风险解释与证据链", expanded=True):
        st.subheader("第一部分｜数据来源")
        source_col1, source_col2 = st.columns(2)
        source_col1.info(
            "**✓ 结构健康监测数据**\n\n获取构件受力状态变化信息"
        )
        source_col2.info(
            "**✓ 裂缝及损伤变化数据**\n\n识别损伤发展趋势"
        )
        source_col3, source_col4 = st.columns(2)
        source_col3.info(
            "**✓ BIM构件属性**\n\n提供材料、尺寸、服役信息"
        )
        source_col4.info(
            "**✓ 历史工程案例**\n\n辅助风险模式匹配"
        )

        st.divider()
        st.subheader("第二部分｜AI分析过程")
        process_col1, process_col2, process_col3 = st.columns(3)
        process_col1.markdown(
            "**1. 状态识别**\n\n根据当前监测数据判断构件健康状态。"
        )
        process_col2.markdown(
            "**2. 趋势预测**\n\n分析未来180天健康变化趋势。"
        )
        process_col3.markdown(
            "**3. 风险关联分析**\n\n结合多源信息识别潜在风险因素。"
        )

        st.divider()
        st.subheader("第三部分｜风险判断与工程建议")
        st.warning(f"**当前风险：** {risk_level}")
        st.markdown(
            """
**主要原因：** 存在持续退化趋势，需要关注后续变化。

**工程建议：** 建议开展针对性检查，并结合实际情况制定维护措施。
            """
        )
        st.success("AI提供辅助决策依据，最终维修方案由工程人员确认。")

    # 以下模块只解释已有数据与推演结果，不参与健康指数和方案计算
    with st.expander("🤖 AI风险解释与决策依据", expanded=True):
        st.subheader("1. 结构异常特征识别")

        initial_data = monitoring_data.iloc[0]

        # 根据初始值和当前模拟值，用简单文字描述指标变化方向
        def describe_change(initial_value, current_value):
            if current_value > initial_value:
                return "持续增长"
            if current_value < initial_value:
                return "有所下降"
            return "基本稳定"

        feature_data = pd.DataFrame(
            {
                "指标": ["裂缝宽度", "应变", "挠度"],
                "初始值": [
                    f"{initial_data['裂缝宽度（mm）']:.2f} mm",
                    f"{initial_data['应变（με）']:.0f} με",
                    f"{initial_data['挠度（mm）']:.1f} mm",
                ],
                "当前模拟值": [
                    f"{simulated_crack:.2f} mm",
                    f"{simulated_strain:.0f} με",
                    f"{simulated_deflection:.1f} mm",
                ],
                "变化趋势": [
                    describe_change(initial_data["裂缝宽度（mm）"], simulated_crack),
                    describe_change(initial_data["应变（με）"], simulated_strain),
                    describe_change(initial_data["挠度（mm）"], simulated_deflection),
                ],
            }
        )
        st.dataframe(feature_data, use_container_width=True, hide_index=True)
        st.info("系统通过多源结构响应指标变化识别潜在风险特征，而非依据单一指标判断。")

        st.subheader("2. 风险原因分析")
        cause_data = pd.DataFrame(
            {
                "原因": ["承载性能退化", "材料老化", "环境因素影响"],
                "匹配程度": ["85%", "45%", "30%"],
                "判断依据": [
                    "裂缝增长、应变升高、挠度增加同步出现",
                    "长期性能下降可能导致刚度降低",
                    "需要结合温湿度数据进一步判断",
                ],
            }
        )
        st.dataframe(cause_data, use_container_width=True, hide_index=True)
        st.caption("说明：候选原因及匹配程度为原型阶段的模拟解释，不代表真实工程诊断结论。")

        st.subheader("3. AI风险解释")
        increased_count = sum(
            [
                simulated_crack > initial_data["裂缝宽度（mm）"],
                simulated_strain > initial_data["应变（με）"],
                simulated_deflection > initial_data["挠度（mm）"],
            ]
        )
        if increased_count >= 2:
            risk_explanation = (
                "综合分析当前裂缝扩展、应变增长以及挠度变化趋势，系统判断当前风险更接近"
                "结构承载性能退化，而非单纯表面裂缝问题。"
            )
        elif increased_count == 1:
            risk_explanation = (
                "当前仅部分结构响应指标出现增长，尚不足以判断为整体承载性能退化，"
                "建议继续监测并重点复核异常指标。"
            )
        else:
            risk_explanation = (
                "当前多项结构响应指标保持稳定或有所下降，暂未识别到明显的同步恶化特征，"
                "建议维持常规巡检和状态跟踪。"
            )
        st.info(risk_explanation)

        st.subheader("4. 维修建议依据")
        decision_evidence = pd.DataFrame(
            {
                "方案": ["不维修", "小修", "大修"],
                "180天后健康指数": [
                    f"{scenarios['情景A：不维修'][-1]:.1f}分",
                    f"{scenarios['情景B：小修方案'][-1]:.1f}分",
                    f"{scenarios['情景C：大修方案'][-1]:.1f}分",
                ],
            }
        )
        st.dataframe(decision_evidence, use_container_width=True, hide_index=True)
        st.success(generate_ai_recommendation(scenarios))

    st.header("7. 🛠️ 维修决策建议")
    st.markdown("**决策依据**")
    basis_col1, basis_col2, basis_col3 = st.columns(3)
    basis_col1.info("**安全性**\n\n未来健康指数变化")
    basis_col2.info("**经济性**\n\n维修投入影响")
    basis_col3.info("**施工影响**\n\n方案实施难度")
    st.caption("数据 → 分析 → 建议")
    recommendation = generate_ai_recommendation(scenarios)
    st.success(recommendation)
    st.warning("提示：本建议由预设规则和模拟数据生成，实际工程决策应由专业结构工程师复核。")

    with st.expander("模型解释与推演可信度说明"):
        st.markdown(
            """
### 1. 状态感知层

系统基于多源结构响应指标进行状态描述。

**输入指标：**

- 裂缝宽度
- 应变
- 挠度

多指标融合能够从不同角度反映结构状态，避免依赖单一指标进行判断。

---

### 2. 健康评价层

通过加权评价模型，将结构响应转换为 **0～100分健康指数**。

**指标权重：**

- 裂缝宽度：40%
- 应变：35%
- 挠度：25%

---

### 3. 趋势预测层

根据当前结构健康状态和指标变化趋势，预测未来180天的健康演化过程。

---

### 4. 反事实推演层

系统通过构造不同维修假设，模拟不同决策路径下的未来状态。

- **不维修：** 保持当前损伤演化趋势。
- **小修：** 通过裂缝修补和局部加固降低损伤发展速度。
- **大修：** 通过整体加固提高结构安全储备。

---

### 5. 决策层

最终方案推荐综合考虑：

- 健康状态
- 安全提升效果
- 维修成本
- 施工影响

在上述证据链基础上形成辅助维修决策。
            """
        )

    st.divider()
    st.subheader("系统能力总结")
    summary_col1, summary_col2 = st.columns(2)
    summary_col1.markdown(
        """
**本系统实现：**

✓ 结构状态感知  
✓ 健康趋势预测  
✓ 维修策略反事实推演  
✓ 智能辅助决策
        """
    )
    summary_col2.markdown(
        """
**未来可扩展：**

- BIM模型接入
- IoT实时监测
- AI大模型增强分析
        """
    )


if __name__ == "__main__":
    main()
