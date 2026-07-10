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
    page_title="青少年脊柱弯曲异常运动疗法疗效效果预测与方案推荐",
    layout="wide"
)


# ============ 2. 当前模型信息 ============
FEATURE_ORDER = ASSETS["model_matrix_columns"]


# ============ 4. 页面标题和说明 + 输入框样式 ============
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f4fbfb 0%, #ffffff 34%);
}
.block-container {
    max-width: 1280px;
    padding-top: 4.5rem;
    padding-bottom: 2rem;
}
.page-title {
    color: #153b45;
    font-size: clamp(1.35rem, 2vw, 1.9rem);
    line-height: 1.32;
    font-weight: 750;
    margin: 0.6rem 0 1.25rem 0;
}
h1 { font-size: 1.25rem !important; }
h3 { font-size: 1.0rem !important; }
input[type="number"], input[type="text"] {
    background-color: #ffffff !important;
    border: 1px solid #d7e2e8 !important;
}
input[type="number"]:placeholder-shown, input[type="text"]:placeholder-shown {
    background-color: #fff1f2 !important;
    border: 1px solid #ef4444 !important;
}
[data-testid="stForm"] {
    background: #ffffff;
    border: 1px solid #d7e7ea;
    border-radius: 14px;
    padding: 1.35rem 1.45rem;
    box-shadow: 0 8px 24px rgba(21, 59, 69, 0.06);
}
div.stButton > button {
    border-radius: 8px;
    border: 1px solid #1f6f8b;
    color: #ffffff;
    background: #1f6f8b;
}
label p {
    font-size: 0.88rem !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="page-title">青少年脊柱弯曲异常运动疗法效果预测与方案推荐系统</div>',
    unsafe_allow_html=True,
)

st.markdown("""
这个网页计算器的总体目的是为了辅助选择合适的运动疗法。通过输入患者的基本信息、体成分和脊柱健康数据，它能即时输出适合该患者的运动疗法和有效改善率。其设计理念是将复杂的多维特征输入转化为直观的个体化预测结果，使“黑箱”模型转化为辅助决策工具，从而推动实际运动疗法方案的指定，为个体化干预决策提供科学依据，实现制定个性化康复方案的目标。
""")

st.info("网页默认提供一组示例数据，用户可直接修改；所有标星项目均为必填。公开版计算器不启用缺失值自动插补；缺失值或超出临床合理范围的数值将被标红，输入不完整或不合理时不会输出预测结果。")

st.divider()


def default_num(var: str, fallback: float = 0.0) -> float:
    try:
        return float(ASSETS["defaults"].get(var, fallback))
    except Exception:
        return fallback


def required_label(text: str) -> str:
    return f"{text} :red[*]"


def example_int(var: str, fallback: int = 0) -> int:
    return int(round(default_num(var, fallback)))

def required_float(value, label: str) -> float:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise ValueError(f"{label} 为必填项，请完整输入。")
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"{label} 必须为数字。") from exc


CLINICAL_RANGES = {
    "height": (1.20, 2.10, "m"),
    "weight": (25.0, 120.0, "kg"),
    "fat_mass": (1.0, 60.0, "kg"),
    "muscle_mass": (10.0, 70.0, "kg"),
    "ati": (0.0, 20.0, "°"),
    "ka": (0.0, 70.0, "°"),
    "balance": (0.0, 10.0, "°"),
    "bmi": (10.0, 45.0, "kg/m²"),
    "fmr": (0.05, 2.0, ""),
}

ACTIVITY_RANGES = {
    "C.LLB": (0.0, 90.0, "°"),
    "C.RLB": (0.0, 90.0, "°"),
    "T.LLB": (0.0, 90.0, "°"),
    "T.RLB": (0.0, 90.0, "°"),
    "L.LLB": (0.0, 90.0, "°"),
    "L.RLB": (0.0, 90.0, "°"),
    "C.FFT": (0.0, 120.0, "°"),
    "C.BF": (0.0, 100.0, "°"),
    "T.FFT": (0.0, 160.0, "°"),
    "T.BF": (0.0, 100.0, "°"),
    "L.FFT": (0.0, 200.0, "°"),
    "L.BF": (0.0, 120.0, "°"),
    "C.LHR": (0.0, 120.0, "°"),
    "C.RHR": (0.0, 120.0, "°"),
    "T.LHR": (0.0, 110.0, "°"),
    "T.RHR": (0.0, 110.0, "°"),
    "L.LHR": (0.0, 110.0, "°"),
    "L.RHR": (0.0, 110.0, "°"),
}


def clinical_range_for_activity(var: str) -> tuple[float, float, str]:
    return ACTIVITY_RANGES.get(var, (0.0, 200.0, "°"))


def _is_blank(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _fmt_num(value: float) -> str:
    return f"{value:g}"


def _fmt_range(min_value: float, max_value: float, unit: str = "") -> str:
    suffix = f" {unit}" if unit else ""
    return f"{_fmt_num(min_value)}-{_fmt_num(max_value)}{suffix}"


def _css_attr_prefix(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def highlight_invalid_inputs(labels: set[str]) -> None:
    if not labels:
        return
    rules = []
    for label in sorted(labels):
        prefix = _css_attr_prefix(label)
        rules.append(
            f'input[aria-label^="{prefix}"] {{ '
            'border: 2px solid #ef4444 !important; '
            'background-color: #fff1f2 !important; '
            'box-shadow: 0 0 0 1px rgba(239, 68, 68, 0.25) !important; '
            '}'
        )
    st.markdown("<style>" + "\n".join(rules) + "</style>", unsafe_allow_html=True)


def check_required_float(
    value,
    label: str,
    min_value: float,
    max_value: float,
    unit: str,
    missing_items: list[str],
    format_items: list[str],
    range_items: list[str],
    invalid_labels: set[str],
) -> float | None:
    if _is_blank(value):
        missing_items.append(label)
        invalid_labels.add(label)
        return None
    try:
        number = float(value)
    except Exception:
        format_items.append(f"{label}：请输入数字")
        invalid_labels.add(label)
        return None
    if number < min_value or number > max_value:
        range_items.append(
            f"{label}：当前值 {_fmt_num(number)}，临床合理范围 {_fmt_range(min_value, max_value, unit)}"
        )
        invalid_labels.add(label)
        return None
    return number


def check_derived_range(
    label: str,
    value: float,
    min_value: float,
    max_value: float,
    unit: str,
    source_labels: list[str],
    range_items: list[str],
    invalid_labels: set[str],
) -> None:
    if value < min_value or value > max_value:
        range_items.append(
            f"{label}：计算值 {_fmt_num(value)}，临床合理范围 {_fmt_range(min_value, max_value, unit)}"
        )
        invalid_labels.update(source_labels)


def show_validation_errors(
    missing_items: list[str],
    format_items: list[str],
    range_items: list[str],
    blocking_items: list[str],
) -> None:
    st.error("无法计算：输入信息不完整或超出临床合理范围。请根据下列提示修改红色边框字段。")
    if missing_items:
        st.markdown("**未填写字段：**")
        for item in missing_items:
            st.write(f"- {item}")
    if format_items:
        st.markdown("**格式错误字段：**")
        for item in format_items:
            st.write(f"- {item}")
    if range_items:
        st.markdown("**超出临床合理范围：**")
        for item in range_items:
            st.write(f"- {item}")
    if blocking_items:
        st.markdown("**其他无法计算原因：**")
        for item in blocking_items:
            st.write(f"- {item}")


DEFAULT_HEIGHT = 1.60
DEFAULT_MUSCLE_MASS = 30.0
DEFAULT_WEIGHT = round(default_num("BMI", 19.69) * DEFAULT_HEIGHT ** 2, 1)
DEFAULT_FAT_MASS = round(default_num("FMR", 0.26) * DEFAULT_MUSCLE_MASS, 1)
AGE_OPTIONS = list(range(13, 19))
DEFAULT_AGE = example_int("Age", 16)
DEFAULT_AGE_INDEX = AGE_OPTIONS.index(DEFAULT_AGE) if DEFAULT_AGE in AGE_OPTIONS else 0
DEFAULT_GENDER_INDEX_CN = 1 if ASSETS["defaults"].get("Gender") == "Female" else 0


def option_index(option_labels, selected_value, mapping):
    for i, label in enumerate(option_labels):
        if mapping[label] == selected_value:
            return i
    return 0


# ============ 5. 输入表单 ============
with st.form("patient_form"):
    st.subheader("① 基本信息")
    col1, col2 = st.columns(2)
    with col1:
        gender_cn = st.selectbox(required_label("性别"), ["男", "女"], index=DEFAULT_GENDER_INDEX_CN)
    with col2:
        age = st.selectbox(required_label("年龄（岁）"), AGE_OPTIONS, index=DEFAULT_AGE_INDEX)

    st.subheader("② 体成分与脊柱形态指标")
    col3, col4, col5, col6 = st.columns(4)
    with col3:
        height = st.text_input(required_label("身高（m）"), value=f"{DEFAULT_HEIGHT:.1f}", placeholder=" ")
    with col4:
        weight = st.text_input(required_label("体重（kg）"), value=f"{DEFAULT_WEIGHT:.1f}", placeholder=" ")
    with col5:
        fat_mass = st.text_input(required_label("体脂量（kg）"), value=f"{DEFAULT_FAT_MASS:.1f}", placeholder=" ")
    with col6:
        muscle_mass = st.text_input(required_label("肌肉量（kg）"), value=f"{DEFAULT_MUSCLE_MASS:.1f}", placeholder=" ")

    col7, col8, col9, col10 = st.columns(4)
    with col7:
        ati = st.text_input(required_label("躯干倾斜角 ATI（°）"), value=str(example_int("ATI", 5)), placeholder=" ")
    with col8:
        ka = st.text_input(required_label("脊柱后凸角 KA（°）"), value=str(example_int("KA", 33)), placeholder=" ")
    with col9:
        st_label = st.selectbox(
            required_label("脊柱侧弯程度 ST"),
            list(ST_OPTIONS_CN.keys()),
            index=option_index(list(ST_OPTIONS_CN.keys()), ASSETS["defaults"].get("ST", "1degree"), ST_OPTIONS_CN)
        )
    with col10:
        sct_label = st.selectbox(
            required_label("脊柱矢状面曲度异常类型（Sagittal curvature abnormality type, SCT）"),
            list(SCT_OPTIONS_CN.keys()),
            index=option_index(list(SCT_OPTIONS_CN.keys()), ASSETS["defaults"].get("SCT", "No"), SCT_OPTIONS_CN)
        )

    st.subheader("③ 脊柱活动度指标（°）")
    act_inputs = {}
    cols_act = st.columns(4)
    for i, var in enumerate(activity_vars()):
        label = ACT_LABEL_MAP_CN.get(var, var)
        with cols_act[i % 4]:
            act_inputs[var] = st.text_input(
                required_label(label), value=str(example_int(var, 0)), placeholder=" ", key=f"act_{var}"
            )

    st.subheader("④ 脊柱平衡度指标（°）")
    bal_inputs = {}
    cols_bal = st.columns(3)
    for i, var in enumerate(balance_vars()):
        label = BAL_LABEL_MAP_CN.get(var, var)
        with cols_bal[i % 3]:
            bal_inputs[var] = st.text_input(
                required_label(label), value=str(example_int(var, 0)), placeholder=" ", key=f"bal_{var}"
            )

    submitted = st.form_submit_button("▶ 计算两种疗法的预测结果与推荐方案")


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
    gender = 0 if gender_cn == "男" else 1
    gender_model = "Male" if gender == 0 else "Female"
    age = int(age)
    st_value = ST_OPTIONS_CN[st_label]
    sct_value = SCT_OPTIONS_CN[sct_label]

    missing_items: list[str] = []
    format_items: list[str] = []
    range_items: list[str] = []
    blocking_items: list[str] = []
    invalid_labels: set[str] = set()

    if st_value == "No" and sct_value == "No":
        blocking_items.append("请至少选择一种脊柱弯曲异常类型：ST 或 SCT 不能同时为无异常。")

    height_value = check_required_float(height, "身高（m）", *CLINICAL_RANGES["height"], missing_items, format_items, range_items, invalid_labels)
    weight_value = check_required_float(weight, "体重（kg）", *CLINICAL_RANGES["weight"], missing_items, format_items, range_items, invalid_labels)
    fat_mass_value = check_required_float(fat_mass, "体脂量（kg）", *CLINICAL_RANGES["fat_mass"], missing_items, format_items, range_items, invalid_labels)
    muscle_mass_value = check_required_float(muscle_mass, "肌肉量（kg）", *CLINICAL_RANGES["muscle_mass"], missing_items, format_items, range_items, invalid_labels)
    ati_value = check_required_float(ati, "躯干倾斜角 ATI（°）", *CLINICAL_RANGES["ati"], missing_items, format_items, range_items, invalid_labels)
    ka_value = check_required_float(ka, "脊柱后凸角 KA（°）", *CLINICAL_RANGES["ka"], missing_items, format_items, range_items, invalid_labels)

    act_values = {}
    for var, value in act_inputs.items():
        label = ACT_LABEL_MAP_CN.get(var, var)
        act_values[var] = check_required_float(value, label, *clinical_range_for_activity(var), missing_items, format_items, range_items, invalid_labels)

    bal_values = {}
    for var, value in bal_inputs.items():
        label = BAL_LABEL_MAP_CN.get(var, var)
        bal_values[var] = check_required_float(value, label, *CLINICAL_RANGES["balance"], missing_items, format_items, range_items, invalid_labels)

    bmi_value = None
    if height_value is not None and weight_value is not None and height_value > 0:
        bmi_value = weight_value / (height_value ** 2)
        check_derived_range("BMI（由身高和体重计算）", bmi_value, *CLINICAL_RANGES["bmi"], ["身高（m）", "体重（kg）"], range_items, invalid_labels)

    fmr_value = None
    if fat_mass_value is not None and muscle_mass_value is not None and muscle_mass_value > 0:
        fmr_value = fat_mass_value / muscle_mass_value
        check_derived_range("FMR（由体脂量和肌肉量计算）", fmr_value, *CLINICAL_RANGES["fmr"], ["体脂量（kg）", "肌肉量（kg）"], range_items, invalid_labels)

    has_validation_errors = bool(missing_items or format_items or range_items or blocking_items)
    if has_validation_errors:
        highlight_invalid_inputs(invalid_labels)
        show_validation_errors(missing_items, format_items, range_items, blocking_items)
        st.stop()

    def predict_one(therapy: str):
        raw = build_raw_row(
            therapy=therapy,
            gender=gender_model,
            age=age,
            bmi=bmi_value,
            fmr=fmr_value,
            st_value=st_value,
            sct_value=sct_value,
            ati=ati_value,
            ka=ka_value,
            act_inputs=act_values,
            bal_inputs=bal_values,
        )
        return predict_patient(raw)

    try:
        res_sps = predict_one("SPS")
        res_combo = predict_one("COM")
        y_sps, p_sps = res_sps["predicted_class"], res_sps["probability"]
        y_combo, p_combo = res_combo["predicted_class"], res_combo["probability"]

        bmi = float(res_combo["complete_raw"]["BMI"])
        fmr = float(res_combo["complete_raw"]["FMR"])
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
                <td style="padding:4px 10px;">BMI（由体重/身高²计算）</td>
                <td style="padding:4px 10px;">{bmi_tag}</td>
              </tr>
              <tr>
                <td style="padding:4px 10px;">FMR（由体脂量/肌肉量计算）</td>
                <td style="padding:4px 10px;">{fmr_tag}</td>
              </tr>
            </table>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()
        st.subheader("两种运动疗法的预测结果对比")

        res_df = pd.DataFrame([
            {
                "疗法": "螺旋肌肉链训练法",
                "预测结局": "改善有效" if y_sps == "Yes" else "改善无效",
                "改善有效概率(%)": round(p_sps * 100, 1),
            },
            {
                "疗法": "螺旋肌肉链训练联合本体感觉神经肌肉促进技术法",
                "预测结局": "改善有效" if y_combo == "Yes" else "改善无效",
                "改善有效概率(%)": round(p_combo * 100, 1),
            },
        ])
        st.dataframe(res_df, use_container_width=True)

        delta = abs(p_sps - p_combo)
        if delta < 0.02:
            st.warning(
                f"两种疗法预测有效概率非常接近（差值 {delta*100:.1f}%），"
                f"建议结合临床经验综合判断。"
            )
        else:
            if p_sps > p_combo:
                st.success(
                    f"推荐方案：**螺旋肌肉链训练法（SPS）**，"
                    f"预测“改善有效”概率约为 **{p_sps*100:.1f}%**。"
                )
            else:
                st.success(
                    f"推荐方案：**螺旋肌肉链训练联合本体感觉神经肌肉促进技术法（COM）**，"
                    f"预测“改善有效”概率约为 **{p_combo*100:.1f}%**。"
                )

        with st.expander("查看本次计算得到的主成分（PC1）："):
            pc = res_combo["pc_values"]
            st.write(pd.DataFrame([{
                "Activity_PC1": pc.get("Activity_PC1"),
                "Activity_PC2": pc.get("Activity_PC2"),
                "Activity_PC3": pc.get("Activity_PC3"),
                "Balance_PC1": pc.get("Balance_PC1"),
                "PC1": pc.get("PC1"),
            }]))

    except Exception as e:
        st.error("无法计算：模型运行出错，请确认部署文件与模型文件一致。")
        st.write(str(e))