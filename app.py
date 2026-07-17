"""B205梁结构健康智能推演系统 V1。"""

from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="B205梁结构健康智能推演系统",
    page_icon="🏗️",
    layout="wide",
)


def set_chinese_font():
    """设置常见中文字体，避免图表中的中文显示为方框。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
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


def predict_health_trend(monitoring_data, days=180):
    """根据当前健康指数和监测数据预测无维修健康趋势。"""
    current_health = get_current_health(monitoring_data)
    daily_rate = calculate_degradation_rate(monitoring_data)
    future_days = np.arange(days + 1)
    future_dates = [date.today() + timedelta(days=int(day)) for day in future_days]

    # 随时间增加轻微加速项，模拟结构自然老化
    health = current_health - daily_rate * future_days - 0.00005 * future_days**2
    return future_dates, np.clip(health, 0, 100)


def simulate_counterfactuals(monitoring_data, days=180):
    """通过改变退化速度，模拟三种维修方案的未来结果。"""
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
    ax.axhline(80, color="#F1C40F", linestyle="--", alpha=0.8, label="关注线（80分）")
    ax.set_xlabel("时间")
    ax.set_ylabel("健康指数")
    ax.set_ylim(max(0, min(health) - 5), min(100, max(health) + 5))
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

    all_values = np.concatenate(list(scenarios.values()))
    ax.set_xlabel("时间")
    ax.set_ylabel("健康指数")
    ax.set_ylim(max(0, min(all_values) - 5), min(100, max(all_values) + 5))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def main():
    """页面主程序。"""
    set_chinese_font()

    st.title("🏗️ B205梁结构健康智能推演系统")
    st.caption("基于AI预测与反事实推演的结构维修决策验证原型")
    st.info("当前为 V1 模拟验证版本，未调用真实AI模型。")

    st.header("1. 构件基础信息")
    col1, col2, col3 = st.columns(3)
    col1.metric("构件编号", "B-2-05")
    col2.metric("类型", "钢筋混凝土梁")
    col3.metric("位置", "教学楼二层")

    # 先生成监测数据，再由最新一条数据计算当前健康指数
    monitoring_data = generate_monitoring_data()
    current_health = get_current_health(monitoring_data)
    risk_level = get_risk_level(current_health)

    st.header("2. 当前结构状态")
    status_col1, status_col2 = st.columns(2)
    status_col1.metric("健康指数", f"{current_health:.1f}%")
    status_col2.metric("风险等级", risk_level)

    st.header("3. 模拟结构监测数据")
    st.dataframe(
        monitoring_data.sort_values("时间", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("说明：以上为最近30天的模拟监测数据，仅用于验证原型流程。")

    st.header("4. 未来180天健康趋势")
    future_dates, health = predict_health_trend(monitoring_data)
    trend_figure = draw_health_chart(future_dates, health)
    st.pyplot(trend_figure, use_container_width=True)
    plt.close(trend_figure)
    st.write(f"预计180天后健康指数约为 **{health[-1]:.1f}分**。")

    st.header("5. 反事实推演")
    st.write("对比不同维修方案通过改变结构退化速度，对未来180天健康状态产生的影响。")
    scenario_dates, scenarios = simulate_counterfactuals(monitoring_data)
    scenario_figure = draw_counterfactual_chart(scenario_dates, scenarios)
    st.pyplot(scenario_figure, use_container_width=True)
    plt.close(scenario_figure)

    result_data = pd.DataFrame(
        {
            "方案": list(scenarios.keys()),
            "180天后健康指数": [f"{values[-1]:.1f}分" for values in scenarios.values()],
            "特点": [
                "无施工成本，但保持原退化速度",
                "投入适中，退化速度降低约55%",
                "安全保持效果最好，退化速度降低约80%",
            ],
        }
    )
    st.dataframe(result_data, use_container_width=True, hide_index=True)

    st.header("6. AI决策建议")
    st.success(
        "综合考虑安全性、经济性和施工影响，推荐采用“小修方案（裂缝修补 + 局部加固）”。"
        "该方案能显著降低结构退化速度，同时避免大修带来的较高成本和较大施工影响。"
    )
    st.warning("提示：本建议由预设规则和模拟数据生成，实际工程决策应由专业结构工程师复核。")


if __name__ == "__main__":
    main()
