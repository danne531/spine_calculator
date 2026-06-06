import streamlit as st
import pandas as pd

from rf708_calculator_core import (
    ACT_LABEL_MAP_EN,
    BAL_LABEL_MAP_EN,
    SCT_OPTIONS_EN,
    ST_OPTIONS_EN,
    ASSETS,
    activity_vars,
    balance_vars,
    build_raw_row,
    predict_patient,
)


# ============ 1. Page config ============
st.set_page_config(
    page_title="Prediction of Exercise Therapy Effectiveness and Treatment Plan Recommendation for Adolescents with Abnormal Spinal Curvature",
    layout="wide"
)


FEATURE_ORDER = ASSETS["model_matrix_columns"]


# ============ 4. Title, description & styles ============

st.markdown(
    """
<style>
.stApp {
    background: linear-gradient(180deg, #f4fbfb 0%, #ffffff 34%);
}
.block-container {
    max-width: 1280px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
.page-title {
    color: #153b45;
    font-size: clamp(1.35rem, 2vw, 1.9rem);
    line-height: 1.32;
    font-weight: 750;
    margin: 0.2rem 0 1.25rem 0;
}
h1 { font-size: 1.25rem !important; }
h3 { font-size: 1.0rem !important; }
input[type="number"] {
    background-color: #ffffff !important;
    border: 1px solid #d7e2e8 !important;
}
input[type="number"]:placeholder-shown {
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
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="page-title">System for Predicting Exercise Therapy Effectiveness and Recommending Treatment Plans for Adolescents with Abnormal Spinal Curvature</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
The overall purpose of this web calculator is to assist in choosing the right exercise therapy. By inputting the patient's basic information, body composition, and spinal health data, it can instantly output exercise therapy and effective improvement rates for that patient. Its design concept is to transform complex multi-dimensional feature inputs into intuitive individualized prediction results, so that the "black box" model can be transformed into an auxiliary decision-making tool, so as to promote the designation of actual exercise therapy plans, provide a scientific basis for individualized intervention decision-making, and achieve the goal of formulating personalized rehabilitation programs.
"""
)

st.info(
    "Numeric fields left blank will be imputed using a random forest imputation procedure before prediction. Values entered by the user will remain unchanged."
)

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


def optional_float(value):
    return None if value is None else float(value)


def optional_int(value):
    return None if value is None else int(value)


# ============ 5. Input form ============
with st.form("patient_form"):
    st.subheader("1. Basic information")
    col1, col2 = st.columns(2)
    with col1:
        gender_text = st.selectbox(required_label("Sex"), ["Male", "Female"], index=0)
    with col2:
        age = st.selectbox(required_label("Age (years)"), list(range(13, 19)), index=0)

    st.subheader("2. Body composition and sagittal/coronal alignment")
    col3, col4, col5, col6 = st.columns(4)
    with col3:
        height = st.number_input("Height (m)", 1.0, 2.0, value=None, step=0.1, format="%.1f", placeholder=" ")
    with col4:
        weight = st.number_input("Weight (kg)", 20.0, 120.0, value=None, step=0.1, format="%.1f", placeholder=" ")
    with col5:
        fat_mass = st.number_input("Fat mass (kg)", 1.0, 80.0, value=None, step=0.1, format="%.1f", placeholder=" ")
    with col6:
        muscle_mass = st.number_input("Muscle mass (kg)", 1.0, 80.0, value=None, step=0.1, format="%.1f", placeholder=" ")

    col7, col8, col9, col10 = st.columns(4)
    with col7:
        ati = st.number_input("Angle of trunk inclination (°)", 0, 40, value=None, step=1, format="%d", placeholder=" ")
    with col8:
        ka = st.number_input("Kyphos angle (°)", 0, 90, value=None, step=1, format="%d", placeholder=" ")
    with col9:
        st_label = st.selectbox(required_label("Scoliosis type"), list(ST_OPTIONS_EN.keys()), index=1)
    with col10:
        sct_label = st.selectbox(
            required_label("Sagittal curvature abnormality type (SCT)"),
            list(SCT_OPTIONS_EN.keys()),
            index=0,
        )

    st.subheader("3. Spinal mobility indices (°)")
    act_inputs = {}
    cols_act = st.columns(4)
    for i, var in enumerate(activity_vars()):
        label = ACT_LABEL_MAP_EN.get(var, var)
        with cols_act[i % 4]:
            act_inputs[var] = st.number_input(
                label, value=None, step=1, format="%d",
                placeholder=" ", key=f"act_{var}"
            )

    st.subheader("4. Spinal balance indices (°)")
    bal_inputs = {}
    cols_bal = st.columns(3)
    for i, var in enumerate(balance_vars()):
        label = BAL_LABEL_MAP_EN.get(var, var)
        with cols_bal[i % 3]:
            bal_inputs[var] = st.number_input(
                label, value=None, step=1, format="%d",
                placeholder=" ", key=f"bal_{var}"
            )

    submitted = st.form_submit_button("▶ Calculate predictions and treatment recommendation")


# ============ 6. Prediction logic ============
def categorize_bmi(bmi_value: float, gender: int):
    if gender == 0:
        if bmi_value < 18.5:
            return "Underweight", "#FFA726"
        elif bmi_value < 24.0:
            return "Normal", "#66BB6A"
        elif bmi_value < 28.0:
            return "Overweight", "#EF5350"
        else:
            return "Obese", "#C62828"
    else:
        if bmi_value < 18.0:
            return "Underweight", "#FFA726"
        elif bmi_value < 23.5:
            return "Normal", "#66BB6A"
        elif bmi_value < 27.0:
            return "Overweight", "#EF5350"
        else:
            return "Obese", "#C62828"


if submitted:
    try:
        gender = 0 if gender_text == "Male" else 1
        gender_model = "Male" if gender == 0 else "Female"
        age = int(age)
        st_value = ST_OPTIONS_EN[st_label]
        sct_value = SCT_OPTIONS_EN[sct_label]

        if st_value == "No" and sct_value == "No":
            st.error("Please select at least one spinal curvature abnormality: ST and SCT cannot both be normal.")
            st.stop()

        height_value = optional_float(height)
        weight_value = optional_float(weight)
        fat_mass_value = optional_float(fat_mass)
        muscle_mass_value = optional_float(muscle_mass)
        ati_value = optional_int(ati)
        ka_value = optional_int(ka)

        def predict_one(therapy: str):
            raw = build_raw_row(
                therapy=therapy,
                gender=gender_model,
                age=age,
                height=height_value,
                weight=weight_value,
                fat_mass=fat_mass_value,
                muscle_mass=muscle_mass_value,
                st_value=st_value,
                sct_value=sct_value,
                ati=ati_value,
                ka=ka_value,
                act_inputs=act_inputs,
                bal_inputs=bal_inputs,
            )
            return predict_patient(raw)

        res_sps = predict_one("SPS")
        res_combo = predict_one("COM")
        y_sps, p_sps = res_sps["predicted_class"], res_sps["probability"]
        y_combo, p_combo = res_combo["predicted_class"], res_combo["probability"]

        bmi = float(res_combo["imputed_raw"]["BMI"])
        fmr = float(res_combo["imputed_raw"]["FMR"])
        bmi_cat, bmi_color = categorize_bmi(bmi, gender)
        bmi_tag = (
            f'<span style="background-color:{bmi_color}; '
            f'color:white; padding:2px 8px; border-radius:12px;">'
            f'{bmi:.1f} ({bmi_cat})</span>'
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
                <th style="text-align:left; padding:4px 10px; border-bottom:1px solid #ddd;">Index</th>
                <th style="text-align:left; padding:4px 10px; border-bottom:1px solid #ddd;">Value / Interpretation</th>
              </tr>
              <tr>
                <td style="padding:4px 10px;">BMI (weight / height², calculated or random-forest imputed)</td>
                <td style="padding:4px 10px;">{bmi_tag}</td>
              </tr>
              <tr>
                <td style="padding:4px 10px;">FMR (fat-to-muscle ratio, calculated or random-forest imputed)</td>
                <td style="padding:4px 10px;">{fmr_tag}</td>
              </tr>
            </table>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()
        st.subheader("Predicted outcomes of two exercise therapies")

        outcome_sps = "Effective improvement" if y_sps == "Yes" else "Ineffective"
        outcome_com = "Effective improvement" if y_combo == "Yes" else "Ineffective"

        st.markdown(
            f"""
            <div style="
                border:1px solid #e0e0e0;
                border-radius:8px;
                padding:10px 12px;
                margin-bottom:8px;
                font-size:0.9rem;">
              <table style="width:100%; border-collapse:collapse;">
                <tr style="background-color:#f7f7f7;">
                  <th style="text-align:left; padding:6px 8px;">Therapy</th>
                  <th style="text-align:left; padding:6px 8px;">Predicted outcome</th>
                  <th style="text-align:left; padding:6px 8px;">
                    Probability of effective<br>improvement (%)
                  </th>
                </tr>
                <tr>
                  <td style="padding:6px 8px; border-top:1px solid #eee;">
                    Spiral muscle chain training (SPS)
                  </td>
                  <td style="padding:6px 8px; border-top:1px solid #eee;">
                    {outcome_sps}
                  </td>
                  <td style="padding:6px 8px; border-top:1px solid #eee;">
                    {p_sps*100:.1f}
                  </td>
                </tr>
                <tr style="background-color:#fafafa;">
                  <td style="padding:6px 8px; border-top:1px solid #eee;">
                    SPS + proprioceptive neuromuscular facilitation (COM)
                  </td>
                  <td style="padding:6px 8px; border-top:1px solid #eee;">
                    {outcome_com}
                  </td>
                  <td style="padding:6px 8px; border-top:1px solid #eee;">
                    {p_combo*100:.1f}
                  </td>
                </tr>
              </table>
            </div>
            """,
            unsafe_allow_html=True
        )

        delta = abs(p_sps - p_combo)
        if delta < 0.02:
            st.warning(
                f"The predicted probabilities are very close (difference {delta*100:.1f}%). "
                f"Clinical judgement is recommended for final decision."
            )
        else:
            if p_sps > p_combo:
                st.success(
                    f"Recommended plan: **Spiral muscle chain training (SPS)**. "
                    f"Estimated probability of effective improvement: **{p_sps*100:.1f}%**."
                )
            else:
                st.success(
                    f"Recommended plan: **SPS + proprioceptive neuromuscular facilitation (COM)**. "
                    f"Estimated probability of effective improvement: **{p_combo*100:.1f}%**."
                )

        with st.expander("View principal component (PC1)"):
            pc = res_combo["pc_values"]
            st.write(pd.DataFrame([{
                "Activity_PC1": pc.get("Activity_PC1"),
                "Activity_PC2": pc.get("Activity_PC2"),
                "Activity_PC3": pc.get("Activity_PC3"),
                "Balance_PC1": pc.get("Balance_PC1"),
                "PC1": pc.get("PC1"),
            }]))

    except Exception as e:
        st.error("An error occurred during prediction. Please check:")
        st.write("- model_assets.json")
        st.write("- rf708_calculator_core.py")
        st.write("- Whether all numeric inputs are reasonable")
        st.exception(e)
