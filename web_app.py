from pathlib import Path
import json
import math
import streamlit as st
import pandas as pd
from rf708_calculator_core import ASSETS, ACT_LABEL_MAP_EN, BAL_LABEL_MAP_EN, activity_vars, balance_vars, derive_categories, predict_patient

st.set_page_config(page_title="Spine Exercise Response Calculator",layout="wide")
E=json.loads(Path(__file__).with_name("example.json").read_text(encoding="utf-8"))
E["Height"] = round(E["Height"] * 100, 1)
for key in E.keys() - {"Gender", "ATI", "KA"}:
    E[key] = float(E[key])
lang=st.sidebar.radio("Language / 语言",["English","中文"])
cn=lang=="中文"
def tr(en,zh): return zh if cn else en
def reset():
    for k,v in E.items():st.session_state[k]=v
for k,v in E.items():
    if k not in st.session_state:st.session_state[k]=v
st.title(tr("Exercise Therapy Response Prediction Calculator for Adolescents with Abnormal Spinal Curvature",
            "青少年脊柱弯曲异常运动疗法疗效预测计算器"))
st.caption(tr("Adolescents with spinal curvature abnormalities · ages 13–18 · exploratory research model","青少年脊柱弯曲异常 · 13–18岁 · 探索性研究模型"))
st.caption("RF model · data2 · 2026-11-21")
st.write(tr(
    "This calculator estimates the probability that an adolescent with abnormal spinal curvature "
    "will meet the prespecified improvement criterion after exercise therapy. After the patient's "
    "basic information, body composition measures, and spinal health indicators are entered, "
    "the calculator uses these baseline data to generate an individualized prediction immediately. "
    "The prediction can inform assessment of expected treatment response, risk stratification, and follow-up.",
    "本计算器用于估计青少年脊柱弯曲异常患者在接受运动干预后达到预设改善标准的概率。"
    "输入患者的基本信息、身体成分及脊柱健康指标后，系统将根据这些基线数据即时给出个体化预测结果，"
    "为疗效预期评估、风险分层和随访提供参考。",
))
st.button(tr("Load illustrative example","载入演示示例"),on_click=reset)
invalid = []
measurements = {}
field_colors = {}

def measurement(label, min_value, max_value, step, key, optional=False):
    field = f"measurement_{key.replace('.', '_')}"
    integer = step == 1
    with st.container(key=field):
        number = st.number_input(label, min_value=min_value, max_value=max_value,
                                 step=step, format="%d" if integer else "%.1f",
                                 value=None, key=key)
        valid = (number is not None and math.isfinite(number)
                 and min_value <= number <= max_value
                 and math.isclose(number, round(number, 0 if integer else 1),
                                  rel_tol=0, abs_tol=1e-10))
        field_colors[field] = "#d5f2d6" if valid else "#fbe0e0"
        if not valid:
            if not (optional and number is None):
                invalid.append(key)
                st.caption(tr(f"Enter a value from {min_value:g} to {max_value:g} in the displayed precision.",
                              f"请输入 {min_value:g} 至 {max_value:g} 之间的{'整数' if integer else '数值（最多一位小数）'}。"))
            return math.nan
        measurements[key] = number
        return number
with st.expander(tr("About this example","示例说明")):
    st.write(tr("Illustrative inputs, not a claimed participant: female, age 16, ATI 8°, KA 45°. Categories are Grade II and Kyphosis.","演示输入，不代表某位真实受试者：女性，16岁，ATI 8°，KA 45°；对应Ⅱ级及后凸。"))
left,right=st.columns([2,1],gap="large")
with left:
    st.subheader(tr("1. Participant and body composition","1. 基本信息与身体成分"))
    c1,c2=st.columns(2)
    with c1:
        st.selectbox(tr("Sex","性别"),["Male","Female"],key="Gender")
        measurement(tr("Height (cm)","身高（cm）"),min_value=100.0,max_value=220.0,step=.1,key="Height")
        measurement(tr("Fat mass (kg, optional)","脂肪量（kg，可留空）"),min_value=.1,max_value=80.0,step=.1,key="FatMass",optional=True)
    with c2:
        measurement(tr("Age (years)","年龄（岁）"),min_value=13.0,max_value=18.0,step=.1,key="Age")
        measurement(tr("Weight (kg)","体重（kg）"),min_value=20.0,max_value=150.0,step=.1,key="Weight")
        measurement(tr("Muscle mass (kg)","肌肉量（kg）"),min_value=1.0,max_value=100.0,step=.1,key="MuscleMass")
    st.subheader(tr("2. Spinal curvature assessment","2. 脊柱曲度评估"))
    c1,c2=st.columns(2)
    with c1:ati=measurement(tr("Angle of trunk inclination (ATI, °)","躯干倾斜角（ATI，°）"),min_value=0,max_value=40,step=1,key="ATI")
    with c2:ka=measurement(tr("Kyphosis angle (KA, °)","脊柱后凸角（KA，°）"),min_value=0,max_value=90,step=1,key="KA")
    angles_ready = math.isfinite(ati) and math.isfinite(ka)
    stv,sctv=derive_categories(ati,ka) if angles_ready else (None,None)
    st_codes=["No","1degree","2degree","3degree"];sct_codes=["No","Flat_back","kyphosis"]
    c1,c2=st.columns(2)
    with c1:st.selectbox(tr("Scoliosis screening category","侧弯筛查分类"),st_codes,index=st_codes.index(stv) if angles_ready else None,format_func=lambda z:dict(zip(st_codes,["No screening abnormality","Grade I","Grade II","Grade III"]))[z],disabled=True)
    with c2:st.selectbox(tr("Sagittal curvature type","矢状面曲度类型"),sct_codes,index=sct_codes.index(sctv) if angles_ready else None,format_func=lambda z:{"No":tr("Normal","正常"),"Flat_back":tr("Flat back","平背"),"kyphosis":tr("Kyphosis","后凸")}[z],disabled=True)
    st.caption(tr("Categories update automatically. KA <20°: flat back; KA 20–40°: normal; KA >40°: kyphosis.","类别自动联动：KA<20°为平背，20–40°为正常，>40°为后凸。"))
    st.subheader(tr("3. Spinal mobility (°)","3. 脊柱活动度（°）"))
    cols=st.columns(3)
    for i,k in enumerate(activity_vars()):
        with cols[i%3]:measurement(ACT_LABEL_MAP_EN[k],min_value=0.0,max_value=180.0,step=.1,key=k)
    st.subheader(tr("4. Spinal balance (°)","4. 脊柱平衡（°）"))
    cols=st.columns(3)
    for i,k in enumerate(balance_vars()):
        with cols[i%3]:measurement(BAL_LABEL_MAP_EN[k],min_value=0.0,max_value=30.0,step=.1,key=k)
styles = "\n".join(
    f".st-key-{field} input {{ background-color: {color} !important; }}"
    for field, color in field_colors.items()
)
st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)

with right:
    st.subheader(tr("Estimated treatment response","预计疗效反应"))
    x={**measurements,"Gender":st.session_state.Gender,"ST":stv,"SCT":sctv}
    if "Height" in x:
        x["Height"] /= 100
    x["BMI"] = x["Weight"]/x["Height"]**2 if {"Weight", "Height"} <= x.keys() else math.nan
    fmr_imputed = st.session_state.FatMass is None
    x["FMR"] = (float(ASSETS["zscaler"]["mean"]["FMR"]) if fmr_imputed
                else x["FatMass"]/x["MuscleMass"] if {"FatMass", "MuscleMass"} <= x.keys()
                else math.nan)
    st.metric("BMI", f'{x["BMI"]:.2f}' if math.isfinite(x["BMI"]) else "—")
    st.metric("FMR", f'{x["FMR"]:.3f}' if math.isfinite(x["FMR"]) else "—")
    if fmr_imputed:
        st.caption(tr("Fat mass is missing. Prediction uses the development-set mean FMR.",
                      "脂肪量缺失，预测使用开发集平均 FMR。"))
    if invalid:
        st.error(tr("Complete or correct the required measurements highlighted in red.", "请补齐或修正标红的必填指标。"))
    elif stv=="No" and sctv=="No":
        st.info(tr("Enter at least one qualifying curvature abnormality.","请提供至少一项符合研究定义的脊柱曲度异常。"))
    else:
        output={}
        for therapy in ["SPS","COM"]:
            r=predict_patient({**x,"Type":therapy});output[therapy]=r
            st.metric(therapy,f'{r["probability"]*100:.1f}%')
        st.caption(tr("Probabilities include the saved Platt calibration.","概率已应用保存的Platt校准参数。"))
        with st.expander(tr("View integrated PC1","查看综合PC1")):
            st.dataframe(pd.DataFrame([output["SPS"]["pc_values"]]),hide_index=True)
        st.download_button(tr("Download estimate","下载预测结果"),json.dumps({"input":x,"FMR_imputed":fmr_imputed,"results":output},ensure_ascii=False,indent=2),"spine_response_estimate.json","application/json")
    st.caption(tr("Exploratory research estimates; external validation is pending. Scenario differences do not establish individual treatment benefit.","结果为探索性研究估计，尚待外部验证；两种情境的预测差异不代表已证实的个体治疗获益。"))


