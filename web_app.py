import streamlit as st
import pandas as pd

from rf708_calculator_core import (
    ACT_LABEL_MAP_CN,
    BAL_LABEL_MAP_CN,
    SCT_OPTIONS_CN,
    ST_OPTIONS_CN,
    ASSETS,
    activity_vars,
    balance_vars,
    build_raw_row,
    predict_patient,
)


# ============ 1. 页面配置 ============
st.set_page_config(
    page_title="青少年脊柱弯曲异常运动干预疗效预测计算器",
    layout="wide"
)


# ============ 2. 当前模型信息 ============
FEATURE_ORDER = ASSETS["model_matrix_columns"]


# ============ 4. 页面标题和说明 + 输入框样式 ============
st.title("青少年脊柱弯曲异常运动干预疗效预测计算器")

st.markdown("""
<style>
h1 { font-size: 1.3rem !important; }
h3 { font-size: 1.0rem !important; }
input[type="number"] { background-color: rgba(255,255,255,1); }
input[type="number"][value]:not([value="0.0"]) {
    background-color: rgba(144,238,144,0.3);
}
.metric-card {
    border: 1px solid #e6e8eb;
    border-radius: 8px;
    padding: 12px 14px;
    background: #ffffff;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
本计算器用于估计患者接受运动干预后达到预设改善标准的概率。录入患者基础信息、体成分以及脊柱健康指标后，工具可即时输出该患者经运动疗法干预后达到预设改善标准的概率。本工具设计理念是将多维基线测量数据转化为个体化模型预测结果，使“黑箱”模型转变为辅助决策工具，用于辅助疗效预期评估、风险分层和追踪随访。
""")

st.divider()


DISPLAY_DEFAULTS = {"C.LLB":45.8,"C.RLB":42.4,"T.LLB":35.3,"T.RLB":37.4,"L.LLB":25.7,"L.RLB":27.3,"C.FFT":76.7,"C.BF":56.4,"T.FFT":73.9,"T.BF":47.4,"L.FFT":132.7,"L.BF":50.6,"C.LHR":74.4,"C.RHR":70.3,"T.LHR":38.4,"T.RHR":35.6,"L.LHR":53.8,"L.RHR":44.6,"HB":2.3,"SB":1.8,"PB":1.6}

def default_num(var: str, fallback: float = 0.0) -> float:
    try:
        return float(DISPLAY_DEFAULTS.get(var, ASSETS["defaults"].get(var, fallback)))
    except Exception:
        return fallback


# ============ 5. 输入表单 ============
with st.form("patient_form"):
    st.subheader("① 基本信息")
    col1, col2 = st.columns(2)
    with col1:
        gender_cn = st.selectbox("性别", ["男", "女"], index=1)
    with col2:
        age = st.number_input(
            "年龄（岁）", min_value=6.0, max_value=25.0,
            value=15.5, step=0.1, format="%.1f"
        )

    st.subheader("② 体成分与脊柱形态指标")
    col3, col4 = st.columns(2)
    with col3:
        height = st.number_input(
            "身高（m）", min_value=1.0, max_value=2.0,
            value=1.6, step=0.1, format="%.1f"
        )
    with col4:
        weight = st.number_input(
            "体重（kg）", min_value=20.0, max_value=120.0,
            value=58.9, step=0.1, format="%.1f"
        )

    col5, col6 = st.columns(2)
    with col5:
        fat_mass = st.number_input(
            "体脂量（kg）", min_value=1.0, max_value=80.0,
            value=7.2, step=0.1, format="%.1f"
        )
    with col6:
        muscle_mass = st.number_input(
            "肌肉量（kg）", min_value=1.0, max_value=80.0,
            value=30.0, step=0.1, format="%.1f"
        )

    col7, col8 = st.columns(2)
    with col7:
        ati = st.number_input(
            "躯干倾斜角 ATI（°）", min_value=0.0, max_value=40.0,
            value=9.5, step=0.1, format="%.1f"
        )
    with col8:
        ka = st.number_input(
            "脊柱后凸角 KA（°）", min_value=0.0, max_value=90.0,
            value=43.0, step=0.1, format="%.1f"
        )

    col9, col10 = st.columns(2)
    with col9:
        st_label = st.selectbox(
            "脊柱侧弯程度 ST",
            list(ST_OPTIONS_CN.keys()),
            index=3
        )
    with col10:
        sct_label = st.selectbox(
            "矢状位弯曲类型 SCT",
            list(SCT_OPTIONS_CN.keys()),
            index=2
        )

    st.subheader("③ 脊柱活动度指标（°）")
    act_inputs = {}
    cols_act = st.columns(4)
    for i, var in enumerate(activity_vars()):
        label = ACT_LABEL_MAP_CN.get(var, var)
        with cols_act[i % 4]:
            act_inputs[var] = st.number_input(
                f"{label}", value=default_num(var), step=0.1, format="%.1f",
                key=f"act_{var}"
            )

    st.subheader("④ 脊柱平衡度指标（°）")
    bal_inputs = {}
    cols_bal = st.columns(4)
    for i, var in enumerate(balance_vars()):
        label = BAL_LABEL_MAP_CN.get(var, var)
        with cols_bal[i % 4]:
            bal_inputs[var] = st.number_input(
                f"{label}", value=default_num(var), step=0.1, format="%.1f",
                key=f"bal_{var}"
            )

    submitted = st.form_submit_button("▶ 计算疗效反应预测概率")


# ============ 6. 预测逻辑 ============
def categorize_bmi(bmi_value: float, gender: int):
    if gender == 0:
        if bmi_value < 18.5:
            return "偏低", "#FFA726"
        elif bmi_value < 24.0:
            return "正常", "#66BB6A"
        elif bmi_value < 28.0:
            return "超重", "#EF5350"
        else:
            return "肥胖", "#C62828"
    else:
        if bmi_value < 18.0:
            return "偏低", "#FFA726"
        elif bmi_value < 23.5:
            return "正常", "#66BB6A"
        elif bmi_value < 27.0:
            return "超重", "#EF5350"
        else:
            return "肥胖", "#C62828"


if submitted:
    try:
        gender = 0 if gender_cn == "男" else 1
        gender_model = "Male" if gender == 0 else "Female"
        st_value = ST_OPTIONS_CN[st_label]
        sct_value = SCT_OPTIONS_CN[sct_label]

        if st_value == "No" and sct_value == "No":
            st.error("请至少选择一种脊柱弯曲异常类型：ST 或 SCT 不能同时为无异常。")
            st.stop()

        if height <= 0:
            st.error("身高必须大于 0，请重新输入。")
            st.stop()
        bmi = weight / (height ** 2)

        if muscle_mass <= 0:
            st.error("肌肉量必须大于 0 才能计算 FMR，请重新输入。")
            st.stop()
        fmr = fat_mass / muscle_mass

        bmi_cat, bmi_color = categorize_bmi(bmi, gender)
        bmi_tag = (
            f'<span style="background-color:{bmi_color}; '
            f'color:white; padding:2px 8px; border-radius:12px;">'
            f'{bmi:.1f}（{bmi_cat}）</span>'
        )
        fmr_tag = (
            f'<span style="background-color:#42A5F5; '
            f'color:white; padding:2px 8px; border-radius:12px;">'
            f'{fmr:.1f}</span>'
        )

        st.markdown(
            f"""
            <div style="margin-top:0.5rem; margin-bottom:0.5rem;">
            <table style="font-size:0.9rem; border-collapse:collapse;">
              <tr>
                <th style="text-align:left; padding:4px 10px; border-bottom:1px solid #ddd;">指标</th>
                <th style="text-align:left; padding:4px 10px; border-bottom:1px solid #ddd;">数值 / 解释</th>
              </tr>
              <tr>
                <td style="padding:4px 10px;">BMI（体重/身高²）</td>
                <td style="padding:4px 10px;">{bmi_tag}</td>
              </tr>
              <tr>
                <td style="padding:4px 10px;">FMR（脂肪肌肉比）</td>
                <td style="padding:4px 10px;">{fmr_tag}</td>
              </tr>
            </table>
            </div>
            """,
            unsafe_allow_html=True
        )

        def predict_one(therapy: str):
            raw = build_raw_row(
                therapy=therapy,
                gender=gender_model,
                age=age,
                bmi=bmi,
                fmr=fmr,
                st_value=st_value,
                sct_value=sct_value,
                ati=ati,
                ka=ka,
                act_inputs=act_inputs,
                bal_inputs=bal_inputs,
            )
            return predict_patient(raw)

        res_sps = predict_one("SPS")
        res_combo = predict_one("COM")
        y_sps, p_sps = res_sps["predicted_class"], res_sps["probability"]
        y_combo, p_combo = res_combo["predicted_class"], res_combo["probability"]

        st.divider()
        st.subheader("探索性疗效反应预测概率")

        res_df = pd.DataFrame([
            {
                "运动疗法情境": "螺旋肌肉链训练法",
                "预测结局": "改善有效" if y_sps == "Yes" else "改善无效",
                "达到改善标准的预测概率（%）": round(p_sps * 100, 1),
            },
            {
                "运动疗法情境": "螺旋肌肉链训练联合本体感觉神经肌肉促进技术法",
                "预测结局": "改善有效" if y_combo == "Yes" else "改善无效",
                "达到改善标准的预测概率（%）": round(p_combo * 100, 1),
            },
        ])
        st.dataframe(res_df, use_container_width=True)

        st.success(
            f"探索性估计：SPS为 **{p_sps*100:.1f}%**，COM为 **{p_combo*100:.1f}%**。"
        )

        with st.expander("查看主成分（PC1）"):
            pc = res_combo["pc_values"]
            st.write(pd.DataFrame([{
                "Activity_PC1": pc.get("Activity_PC1"),
                "Activity_PC2": pc.get("Activity_PC2"),
                "Activity_PC3": pc.get("Activity_PC3"),
                "Balance_PC1": pc.get("Balance_PC1"),
                "PC1": pc.get("PC1"),
            }]))

    except Exception as e:
        st.error("预测时出现错误，请检查以下项目：")
        st.write("- model_assets.json 是否与当前 RF 模型一致")
        st.write("- rf708_calculator_core.py 是否与网页文件放在同一目录")
        st.write("- 输入数值是否完整且合理（尤其是身高、体重、体脂、肌肉量）")
        st.exception(e)
