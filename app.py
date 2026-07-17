"""B205梁结构健康智能推演系统 V1。

健康指数由裂缝宽度、应变和挠度计算，维修方案通过改变退化速度进行推演。
"""

from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="B205梁结构健康智能推演系统",
    page_icon="🏗️",
    layout="wide",
)


def set_chinese_font():
    """自动加载中英文字体，兼容 Windows 和 Streamlit Cloud（Linux）。"""
    # 优先级从云端 Linux 常见字体到 Windows 常见字体
    preferred_fonts = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",  # Noto 的 TTC 字体有时以 JP 名称注册，但包含中文字形
        "Source Han Sans SC",
        "Source Han Sans CN",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
    ]

    # 主动注册常见字体文件。也支持将字体放入项目的 fonts 目录后自动加载。
    project_dir = Path(__file__).resolve().parent
    candidate_paths = [
        project_dir / "fonts" / "NotoSansCJKsc-Regular.otf",
        project_dir / "fonts" / "NotoSansCJK-Regular.ttc",
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJKsc-Regular.otf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]

    for font_path in candidate_paths:
        if font_path.exists():
            try:
                font_manager.fontManager.addfont(str(font_path))
            except (OSError, RuntimeError, ValueError, AttributeError):
                # 某些系统不支持读取特定字体集合，继续尝试其他字体
                pass

    available_names = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = next(
        (font_name for font_name in preferred_fonts if font_name in available_names),
        None,
    )

    # 兼容同一字体在不同 Linux 发行版中的名称差异
    if selected_font is None:
        chinese_font_keywords = ("Noto Sans CJK", "Source Han Sans", "YaHei", "SimHei")
        selected_font = next(
            (
                font_name
                for font_name in sorted(available_names)
                if any(keyword in font_name for keyword in chinese_font_keywords)
            ),
            "DejaVu Sans",
        )

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [selected_font, *preferred_fonts, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


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
        arrowprops={"arrowstyle": "->", "color": "#C0392B"},
    )

    ax.axhline(80, color="#F1C40F", linestyle="--", alpha=0.8, label="关注线（80分）")
    ax.set_title(f"未来180天健康趋势预测（当前健康指数：{current_health:.1f}分）")
    ax.set_xlabel("时间")
    ax.set_ylabel("健康指数")
    # 固定坐标范围，确保不同模拟状态的曲线高低位置可以直接比较
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend()
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
    ax.set_title(f"未来180天反事实推演（当前健康指数：{current_health:.1f}分）")
    ax.set_xlabel("时间")
    ax.set_ylabel("健康指数")
    # 与健康趋势图使用相同的固定范围，避免自动缩放造成视觉误差
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
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
