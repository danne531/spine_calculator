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
    "The webpage provides one default example case that users can edit directly. All starred fields are required. The public calculator does not perform automatic missing-value imputation; missing values or clinically implausible values will be highlighted in red, and predictions are not generated until all inputs are complete and reasonable."
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

def required_float(value, label: str) -> float:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise ValueError(f"{label} is required. Please provide a complete input.")
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"{label} must be numeric.") from exc


CLINICAL_RANGES = {
    "height": (1.20, 2.10, "m"),
    "weight": (25.0, 120.0, "kg"),
    "fat_mass": (1.0, 60.0, "kg"),
    "muscle_mass": (10.0, 70.0, "kg"),
    "ati": (0.0, 30.0, "°"),
    "ka": (0.0, 80.0, "°"),
    "activity": (0.0, 180.0, "°"),
    "balance": (0.0, 30.0, "°"),
    "bmi": (10.0, 45.0, "kg/m²"),
    "fmr": (0.05, 2.0, ""),
}


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
        format_items.append(f"{label}: please enter a numeric value")
        invalid_labels.add(label)
        return None
    if number < min_value or number > max_value:
        range_items.append(
            f"{label}: current value {_fmt_num(number)}, clinically plausible range {_fmt_range(min_value, max_value, unit)}"
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
            f"{label}: calculated value {_fmt_num(value)}, clinically plausible range {_fmt_range(min_value, max_value, unit)}"
        )
        invalid_labels.update(source_labels)


def show_validation_errors(
    missing_items: list[str],
    format_items: list[str],
    range_items: list[str],
    blocking_items: list[str],
) -> None:
    st.error("Unable to calculate: some inputs are missing or outside the clinically plausible range. Please correct the fields highlighted in red.")
    if missing_items:
        st.markdown("**Missing fields:**")
        for item in missing_items:
            st.write(f"- {item}")
    if format_items:
        st.markdown("**Invalid numeric format:**")
        for item in format_items:
            st.write(f"- {item}")
    if range_items:
        st.markdown("**Outside the clinically plausible range:**")
        for item in range_items:
            st.write(f"- {item}")
    if blocking_items:
        st.markdown("**Other reasons why calculation is unavailable:**")
        for item in blocking_items:
            st.write(f"- {item}")


DEFAULT_HEIGHT = 1.60
DEFAULT_MUSCLE_MASS = 30.0
DEFAULT_WEIGHT = round(default_num("BMI", 19.69) * DEFAULT_HEIGHT ** 2, 1)
DEFAULT_FAT_MASS = round(default_num("FMR", 0.26) * DEFAULT_MUSCLE_MASS, 1)
AGE_OPTIONS = list(range(13, 19))
DEFAULT_AGE = example_int("Age", 16)
DEFAULT_AGE_INDEX = AGE_OPTIONS.index(DEFAULT_AGE) if DEFAULT_AGE in AGE_OPTIONS else 0
DEFAULT_GENDER_INDEX_EN = 1 if ASSETS["defaults"].get("Gender") == "Female" else 0


def option_index(option_labels, selected_value, mapping):
    for i, label in enumerate(option_labels):
        if mapping[label] == selected_value:
            return i
    return 0


# ============ 5. Input form ============
with st.form("patient_form"):
    st.subheader("1. Basic information")
    col1, col2 = st.columns(2)
    with col1:
        gender_text = st.selectbox(required_label("Sex"), ["Male", "Female"], index=DEFAULT_GENDER_INDEX_EN)
    with col2:
        age = st.selectbox(required_label("Age (years)"), AGE_OPTIONS, index=DEFAULT_AGE_INDEX)

    st.subheader("2. Body composition and sagittal/coronal alignment")
    col3, col4, col5, col6 = st.columns(4)
    with col3:
        height = st.text_input(required_label("Height (m)"), value=f"{DEFAULT_HEIGHT:.1f}", placeholder=" ")
    with col4:
        weight = st.text_input(required_label("Weight (kg)"), value=f"{DEFAULT_WEIGHT:.1f}", placeholder=" ")
    with col5:
        fat_mass = st.text_input(required_label("Fat mass (kg)"), value=f"{DEFAULT_FAT_MASS:.1f}", placeholder=" ")
    with col6:
        muscle_mass = st.text_input(required_label("Muscle mass (kg)"), value=f"{DEFAULT_MUSCLE_MASS:.1f}", placeholder=" ")

    col7, col8, col9, col10 = st.columns(4)
    with col7:
        ati = st.text_input(required_label("Angle of trunk inclination (°)"), value=str(example_int("ATI", 5)), placeholder=" ")
    with col8:
        ka = st.text_input(required_label("Kyphos angle (°)"), value=str(example_int("KA", 33)), placeholder=" ")
    with col9:
        st_label = st.selectbox(
            required_label("Scoliosis type"),
            list(ST_OPTIONS_EN.keys()),
            index=option_index(list(ST_OPTIONS_EN.keys()), ASSETS["defaults"].get("ST", "1degree"), ST_OPTIONS_EN),
        )
    with col10:
        sct_label = st.selectbox(
            required_label("Sagittal curvature abnormality type (SCT)"),
            list(SCT_OPTIONS_EN.keys()),
            index=option_index(list(SCT_OPTIONS_EN.keys()), ASSETS["defaults"].get("SCT", "No"), SCT_OPTIONS_EN),
        )

    st.subheader("3. Spinal mobility indices (°)")
    act_inputs = {}
    cols_act = st.columns(4)
    for i, var in enumerate(activity_vars()):
        label = ACT_LABEL_MAP_EN.get(var, var)
        with cols_act[i % 4]:
            act_inputs[var] = st.text_input(
                required_label(label), value=str(example_int(var, 0)), placeholder=" ", key=f"act_{var}"
            )

    st.subheader("4. Spinal balance indices (°)")
    bal_inputs = {}
    cols_bal = st.columns(3)
    for i, var in enumerate(balance_vars()):
        label = BAL_LABEL_MAP_EN.get(var, var)
        with cols_bal[i % 3]:
            bal_inputs[var] = st.text_input(
                required_label(label), value=str(example_int(var, 0)), placeholder=" ", key=f"bal_{var}"
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
    gender = 0 if gender_text == "Male" else 1
    gender_model = "Male" if gender == 0 else "Female"
    age = int(age)
    st_value = ST_OPTIONS_EN[st_label]
    sct_value = SCT_OPTIONS_EN[sct_label]

    missing_items: list[str] = []
    format_items: list[str] = []
    range_items: list[str] = []
    blocking_items: list[str] = []
    invalid_labels: set[str] = set()

    if st_value == "No" and sct_value == "No":
        blocking_items.append("Please select at least one spinal curvature abnormality: ST and SCT cannot both be normal.")

    height_value = check_required_float(height, "Height (m)", *CLINICAL_RANGES["height"], missing_items, format_items, range_items, invalid_labels)
    weight_value = check_required_float(weight, "Weight (kg)", *CLINICAL_RANGES["weight"], missing_items, format_items, range_items, invalid_labels)
    fat_mass_value = check_required_float(fat_mass, "Fat mass (kg)", *CLINICAL_RANGES["fat_mass"], missing_items, format_items, range_items, invalid_labels)
    muscle_mass_value = check_required_float(muscle_mass, "Muscle mass (kg)", *CLINICAL_RANGES["muscle_mass"], missing_items, format_items, range_items, invalid_labels)
    ati_value = check_required_float(ati, "Angle of trunk inclination (°)", *CLINICAL_RANGES["ati"], missing_items, format_items, range_items, invalid_labels)
    ka_value = check_required_float(ka, "Kyphos angle (°)", *CLINICAL_RANGES["ka"], missing_items, format_items, range_items, invalid_labels)

    act_values = {}
    for var, value in act_inputs.items():
        label = ACT_LABEL_MAP_EN.get(var, var)
        act_values[var] = check_required_float(value, label, *CLINICAL_RANGES["activity"], missing_items, format_items, range_items, invalid_labels)

    bal_values = {}
    for var, value in bal_inputs.items():
        label = BAL_LABEL_MAP_EN.get(var, var)
        bal_values[var] = check_required_float(value, label, *CLINICAL_RANGES["balance"], missing_items, format_items, range_items, invalid_labels)

    bmi_value = None
    if height_value is not None and weight_value is not None and height_value > 0:
        bmi_value = weight_value / (height_value ** 2)
        check_derived_range("BMI (calculated from height and weight)", bmi_value, *CLINICAL_RANGES["bmi"], ["Height (m)", "Weight (kg)"], range_items, invalid_labels)

    fmr_value = None
    if fat_mass_value is not None and muscle_mass_value is not None and muscle_mass_value > 0:
        fmr_value = fat_mass_value / muscle_mass_value
        check_derived_range("FMR (calculated from fat mass and muscle mass)", fmr_value, *CLINICAL_RANGES["fmr"], ["Fat mass (kg)", "Muscle mass (kg)"], range_items, invalid_labels)

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
                <td style="padding:4px 10px;">BMI (calculated as weight / height²)</td>
                <td style="padding:4px 10px;">{bmi_tag}</td>
              </tr>
              <tr>
                <td style="padding:4px 10px;">FMR (calculated as fat mass / muscle mass)</td>
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
        st.error("Unable to calculate: the model could not run. Please confirm that deployment files and model assets are consistent.")
        st.write(str(e))