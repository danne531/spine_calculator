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
    page_title="Exercise Therapy Response Prediction Calculator for Adolescents with Abnormal Spinal Curvature",
    layout="centered"
)


FEATURE_ORDER = ASSETS["model_matrix_columns"]


# ============ 4. Title, description & styles ============
st.title("Exercise Therapy Response Prediction Calculator for Adolescents with Abnormal Spinal Curvature")

st.markdown(
    """
<style>
h1 { font-size: 1.3rem !important; }
h3 { font-size: 1.0rem !important; }
input[type="number"] { background-color: rgba(255,255,255,1); }
input[type="number"][value]:not([value="0.0"]) {
    background-color: rgba(144,238,144,0.3);
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
This calculator estimates the probability that a patient will achieve the prespecified improvement criterion after exercise intervention. After the patient's baseline information, body-composition measures, and spinal-health indicators are entered, the tool immediately outputs the predicted probability of achieving the prespecified improvement criterion following exercise therapy. The tool converts multidimensional baseline measurements into individualized model predictions, transforming the “black-box” model into a decision-support tool for outcome expectation assessment, risk stratification, and follow-up.
"""
)

st.divider()


DISPLAY_DEFAULTS = {"C.LLB":45.8,"C.RLB":42.4,"T.LLB":35.3,"T.RLB":37.4,"L.LLB":25.7,"L.RLB":27.3,"C.FFT":76.7,"C.BF":56.4,"T.FFT":73.9,"T.BF":47.4,"L.FFT":132.7,"L.BF":50.6,"C.LHR":74.4,"C.RHR":70.3,"T.LHR":38.4,"T.RHR":35.6,"L.LHR":53.8,"L.RHR":44.6,"HB":2.3,"SB":1.8,"PB":1.6}

def default_num(var: str, fallback: float = 0.0) -> float:
    try:
        return float(DISPLAY_DEFAULTS.get(var, ASSETS["defaults"].get(var, fallback)))
    except Exception:
        return fallback


# ============ 5. Input form ============
with st.form("patient_form"):
    st.subheader("1. Basic information")
    col1, col2 = st.columns(2)
    with col1:
        gender_text = st.selectbox("Sex", ["Male", "Female"], index=1)
    with col2:
        age = st.number_input("Age (years)", 6.0, 25.0, 15.5, 0.1, format="%.1f")

    st.subheader("2. Body composition and sagittal/coronal alignment")
    col3, col4 = st.columns(2)
    with col3:
        height = st.number_input("Height (m)", 1.0, 2.0, 1.6, 0.1, format="%.1f")
    with col4:
        weight = st.number_input("Weight (kg)", 20.0, 120.0, 58.9, 0.1, format="%.1f")

    col5, col6 = st.columns(2)
    with col5:
        fat_mass = st.number_input("Fat mass (kg)", 1.0, 80.0, 7.2, 0.1, format="%.1f")
    with col6:
        muscle_mass = st.number_input("Muscle mass (kg)", 1.0, 80.0, 30.0, 0.1, format="%.1f")

    col7, col8 = st.columns(2)
    with col7:
        ati = st.number_input("Angle of trunk inclination (°)", 0.0, 40.0, 9.5, 0.1, format="%.1f")
    with col8:
        ka = st.number_input("Kyphos angle (°)", 0.0, 90.0, 43.0, 0.1, format="%.1f")

    col9, col10 = st.columns(2)
    with col9:
        st_label = st.selectbox("Scoliosis type", list(ST_OPTIONS_EN.keys()), index=3)
    with col10:
        sct_label = st.selectbox("Spinal curvature type", list(SCT_OPTIONS_EN.keys()), index=2)

    st.subheader("3. Spinal mobility indices (°)")
    act_inputs = {}
    cols_act = st.columns(3)
    for i, var in enumerate(activity_vars()):
        label = ACT_LABEL_MAP_EN.get(var, var)
        with cols_act[i % 3]:
            act_inputs[var] = st.number_input(
                f"{label}", value=default_num(var), step=0.1, format="%.1f", key=f"act_{var}"
            )

    st.subheader("4. Spinal balance indices (°)")
    bal_inputs = {}
    cols_bal = st.columns(3)
    for i, var in enumerate(balance_vars()):
        label = BAL_LABEL_MAP_EN.get(var, var)
        with cols_bal[i % 3]:
            bal_inputs[var] = st.number_input(
                f"{label}", value=default_num(var), step=0.1, format="%.1f", key=f"bal_{var}"
            )

    submitted = st.form_submit_button("▶ Calculate predicted probability of treatment response")


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
        st_value = ST_OPTIONS_EN[st_label]
        sct_value = SCT_OPTIONS_EN[sct_label]

        if st_value == "No" and sct_value == "No":
            st.error("Please select at least one spinal curvature abnormality: ST and SCT cannot both be normal.")
            st.stop()

        if height <= 0:
            st.error("Height must be greater than 0. Please check the input.")
            st.stop()
        bmi = weight / (height ** 2)

        if muscle_mass <= 0:
            st.error("Muscle mass must be greater than 0 to compute FMR.")
            st.stop()
        fmr = fat_mass / muscle_mass

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
                <td style="padding:4px 10px;">BMI (weight / height²)</td>
                <td style="padding:4px 10px;">{bmi_tag}</td>
              </tr>
              <tr>
                <td style="padding:4px 10px;">FMR (fat-to-muscle ratio)</td>
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
        st.subheader("Exploratory predicted probabilities of treatment response")

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
                  <th style="text-align:left; padding:6px 8px;">Exercise-therapy scenario</th>
                  <th style="text-align:left; padding:6px 8px;">Predicted outcome</th>
                  <th style="text-align:left; padding:6px 8px;">
                    Predicted probability of meeting<br>the improvement criterion (%)
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

        st.success(
            f"Exploratory estimates: SPS **{p_sps*100:.1f}%**; COM **{p_combo*100:.1f}%**."
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
