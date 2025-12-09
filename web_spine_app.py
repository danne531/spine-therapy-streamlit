import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============ 1. Page config ============
st.set_page_config(
    page_title="Prediction of Exercise Therapy Effectiveness and Treatment Plan Recommendation for Adolescents with Abnormal Spinal Curvature",
    layout="centered"
)

# ============ 2. Load model, imputer & PCA params ============

@st.cache_resource
def load_model():
    model = joblib.load("best_rf_model.pkl")
    return model

@st.cache_resource
def load_imputer():
    imputer = joblib.load("en_rf_imputer.pkl")
    return imputer

@st.cache_resource
def load_pca_params():
    # ---- Mobility PCA ----
    center1_df = pd.read_csv("pca_set1_center.csv")
    scale1_df  = pd.read_csv("pca_set1_scale.csv")
    loadings1_df = pd.read_csv("pca_set1_loadings.csv")

    act_vars = center1_df.iloc[:, 0].tolist()
    center1  = center1_df.iloc[:, 1].to_numpy(dtype=float)
    scale1   = scale1_df.iloc[:, 1].to_numpy(dtype=float)
    rotation1 = (
        loadings1_df
        .set_index(loadings1_df.columns[0])
        .to_numpy(dtype=float)
    )

    # ---- Balance PCA ----
    center2_df = pd.read_csv("pca_set2_center.csv")
    scale2_df  = pd.read_csv("pca_set2_scale.csv")
    loadings2_df = pd.read_csv("pca_set2_loadings.csv")

    bal_vars = center2_df.iloc[:, 0].tolist()
    center2  = center2_df.iloc[:, 1].to_numpy(dtype=float)
    scale2   = scale2_df.iloc[:, 1].to_numpy(dtype=float)
    rotation2 = (
        loadings2_df
        .set_index(loadings2_df.columns[0])
        .to_numpy(dtype=float)
    )

    return {
        "act_vars": act_vars,
        "center1": center1,
        "scale1":  scale1,
        "rotation1": rotation1,
        "bal_vars": bal_vars,
        "center2": center2,
        "scale2":  scale2,
        "rotation2": rotation2,
    }

model = load_model()
imputer = load_imputer()
pca_params = load_pca_params()

act_vars = pca_params["act_vars"]
center1 = pca_params["center1"]
scale1 = pca_params["scale1"]
rotation1 = pca_params["rotation1"]

bal_vars = pca_params["bal_vars"]
center2 = pca_params["center2"]
scale2 = pca_params["scale2"]
rotation2 = pca_params["rotation2"]

FEATURE_ORDER = [
    "Type", "Gender", "Age", "BMI", "FMR",
    "ST", "SCT", "ATI", "KA",
    "PC1", "PC2", "PC3", "PC4"
]

# ======== 2.1 label maps ========

ACT_LABEL_MAP = {
    "C-LLB": "Cervical left lateral bending",
    "C-RLB": "Cervical right lateral bending",
    "T-LLB": "Thoracic left lateral bending",
    "T-RLB": "Thoracic right lateral bending",
    "L-LLB": "Lumbar left lateral bending",
    "L-RLB": "Lumbar right lateral bending",
    "C-FFT": "Cervical flexion",
    "C-BF":  "Cervical extension",
    "T-FFT": "Thoracic flexion",
    "T-BF":  "Thoracic extension",
    "L-FFT": "Lumbar flexion",
    "L-BF":  "Lumbar extension",
    "C-LHR": "Cervical left rotation",
    "C-RHR": "Cervical right rotation",
    "T-LHR": "Thoracic left rotation",
    "T-RHR": "Thoracic right rotation",
    "L-LHR": "Lumbar left rotation",
    "L-RHR": "Lumbar right rotation",
}

BAL_LABEL_MAP = {
    "HB": "Head balance",
    "SB": "Shoulder balance",
    "PB": "Pelvic balance",
}

ST_OPTIONS = {
    "No scoliosis": 0,
    "Grade I scoliosis": 1,
    "Grade II scoliosis": 2,
    "Grade III scoliosis": 3,
}

SCT_OPTIONS = {
    "Normal": 0,
    "Lordosis": 1,
    "Kyphosis": 2,
    "Flat back": 3,
}

# ============ 3. PCA helper functions ============

def compute_activity_pcs_from_r(act_input: dict) -> np.ndarray:
    X_vec = np.array([act_input[var] for var in act_vars], dtype=float).reshape(1, -1)
    X_scaled = (X_vec - center1) / scale1
    PCs = np.dot(X_scaled, rotation1)
    return PCs[0]

def compute_balance_pcs_from_r(bal_input: dict) -> np.ndarray:
    X_vec = np.array([bal_input[var] for var in bal_vars], dtype=float).reshape(1, -1)
    X_scaled = (X_vec - center2) / scale2
    PCs = np.dot(X_scaled, rotation2)
    return PCs[0]

def make_X(type_code, gender, age, bmi, fmr, st_code, sct_code,
           ati, ka, pc1, pc2, pc3, pc4):
    row = {
        "Type": float(type_code),
        "Gender": float(gender),
        "Age": float(age),
        "BMI": float(bmi),
        "FMR": float(fmr),
        "ST": float(st_code),
        "SCT": float(sct_code),
        "ATI": float(ati),
        "KA": float(ka),
        "PC1": float(pc1),
        "PC2": float(pc2),
        "PC3": float(pc3),
        "PC4": float(pc4),
    }
    return pd.DataFrame([[row[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)

# ============ 4. Title, description & styles ============

st.title("System for Predicting Exercise Therapy Effectiveness and Recommending Treatment Plans for Adolescents with Abnormal Spinal Curvature")

st.markdown(
    """
<style>
h1 { font-size: 1.3rem !important; }
h3 { font-size: 1.0rem !important; }
input[type="number"] { background-color: rgba(255,255,255,1); }
/* filled number inputs light green */
input[type="number"][value]:not([value="0.0"]) {
    background-color: rgba(144,238,144,0.3);
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
The overall purpose of this web calculator is to assist in choosing the right exercise therapy. By inputting the patient's basic information, body composition, and spinal health data, it can instantly output exercise therapy and effective improvement rates for that patient. Its design concept is to transform complex multi-dimensional feature inputs into intuitive individualized prediction results, so that the "black box" model can be transformed into an auxiliary decision-making tool, so as to promote the designation of actual exercise therapy plans, provide a scientific basis for individualized intervention decision-making, and achieve the goal of formulating personalized rehabilitation programs.
"""
)

st.divider()

# ============ 5. Input form ============

with st.form("patient_form"):
    st.subheader("1. Basic information")
    col1, col2 = st.columns(2)
    with col1:
        gender_text = st.selectbox("Sex", ["Male", "Female"])
    with col2:
        age = st.number_input("Age (years)", 6.0, 25.0, 13.0, 0.1, format="%.1f")

    st.subheader("2. Body composition and sagittal/coronal alignment")
    col3, col4 = st.columns(2)
    with col3:
        height = st.number_input("Height (m)", 1.0, 2.0, 1.6, 0.1, format="%.1f")
    with col4:
        weight = st.number_input("Weight (kg)", 20.0, 120.0, 50.0, 0.1, format="%.1f")

    col5, col6 = st.columns(2)
    with col5:
        fat_mass = st.number_input("Fat mass (kg)", 1.0, 80.0, 10.0, 0.1, format="%.1f")
    with col6:
        muscle_mass = st.number_input("Muscle mass (kg)", 1.0, 80.0, 30.0, 0.1, format="%.1f")

    col7, col8 = st.columns(2)
    with col7:
        ati = st.number_input("Angle of trunk inclination (°)", 0.0, 40.0, 6.0, 0.1, format="%.1f")
    with col8:
        ka = st.number_input("Kyphos angle (°)", 0.0, 90.0, 35.0, 0.1, format="%.1f")

    col9, col10 = st.columns(2)
    with col9:
        st_label = st.selectbox("Scoliosis type", list(ST_OPTIONS.keys()), index=0)
    with col10:
        sct_label = st.selectbox("Spinal curvature type", list(SCT_OPTIONS.keys()), index=0)

    st.subheader("3. Spinal mobility indices (°)")
    act_inputs = {}
    cols_act = st.columns(3)
    for i, var in enumerate(act_vars):
        label = ACT_LABEL_MAP.get(var, var)
        with cols_act[i % 3]:
            act_inputs[var] = st.number_input(
                f"{label}", value=0.0, step=0.1, format="%.1f", key=f"act_{var}"
            )

    st.subheader("4. Spinal balance indices (°)")
    bal_inputs = {}
    cols_bal = st.columns(3)
    for i, var in enumerate(bal_vars):
        label = BAL_LABEL_MAP.get(var, var)
        with cols_bal[i % 3]:
            bal_inputs[var] = st.number_input(
                f"{label}", value=0.0, step=0.1, format="%.1f", key=f"bal_{var}"
            )

    submitted = st.form_submit_button("▶ Calculate predictions and treatment recommendation")

# ============ 6. Prediction logic ============

def categorize_bmi(bmi_value: float, gender: int):
    if gender == 0:  # male
        if bmi_value < 18.5:
            return "Underweight", "#FFA726"
        elif bmi_value < 24.0:
            return "Normal", "#66BB6A"
        elif bmi_value < 28.0:
            return "Overweight", "#EF5350"
        else:
            return "Obese", "#C62828"
    else:  # female
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
        st_code = ST_OPTIONS[st_label]
        sct_code = SCT_OPTIONS[sct_label]

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

        # BMI & FMR summary table
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

        act_pcs = compute_activity_pcs_from_r(act_inputs)
        bal_pcs = compute_balance_pcs_from_r(bal_inputs)

        if act_pcs.size < 3 or bal_pcs.size < 1:
            st.error("PCA dimension mismatch. Please check exported PCA files.")
            st.stop()

        pc1, pc2, pc3 = act_pcs[0], act_pcs[1], act_pcs[2]
        pc4 = bal_pcs[0]

        def predict_one(type_code: int):
            X_raw = make_X(
                type_code, gender, age, bmi, fmr,
                st_code, sct_code, ati, ka,
                pc1, pc2, pc3, pc4
            )
            X_imp_arr = imputer.transform(X_raw)
            X_imp = pd.DataFrame(X_imp_arr, columns=FEATURE_ORDER)
            y_pred = int(model.predict(X_imp)[0])
            prob_yes = float(model.predict_proba(X_imp)[0, 1])
            return y_pred, prob_yes

        y_sps, p_sps = predict_one(0)
        y_combo, p_combo = predict_one(1)

        # ===== 6.5 Vertical, single-table layout for results =====
        st.divider()
        st.subheader("Predicted outcomes of two exercise therapies")

        outcome_sps = "Effective improvement" if y_sps == 1 else "Ineffective"
        outcome_com = "Effective improvement" if y_combo == 1 else "Ineffective"

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

        # 6.6 Recommendation logic
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

        # 6.7 Show PCs
        with st.expander("View principal components (PC1–PC4)"):
            st.write(pd.DataFrame([{
                "PC1": pc1, "PC2": pc2, "PC3": pc3, "PC4": pc4
            }]))

    except Exception as e:
        st.error("An error occurred during prediction. Please check:")
        st.write("- best_rf_model.pkl")
        st.write("- rf_imputer.pkl")
        st.write("- PCA CSV files (names, centers, scales, loadings)")
        st.write("- Whether all numeric inputs are reasonable")
        st.exception(e)

