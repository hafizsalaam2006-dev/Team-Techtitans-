import streamlit as st
import pandas as pd
import plotly.express as px

from backend.generator import generate_dataset
from backend.anomaly import detect_anomalies
from backend.validator import validate_dataset


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Synthetic Data Generator",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# LIGHT THEME UI
# =========================================================

st.markdown("""
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.stApp {
    background-color: #f8fafc;
    color: #1e293b;
}

/* Main container */

.block-container {
    padding-top: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
    max-width: 1400px;
}

/* Header */

.main-title {
    font-size: 36px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 4px;
}

.subtitle {
    color: #64748b;
    font-size: 16px;
    margin-bottom: 30px;
}

/* Section titles */

.section-title {
    font-size: 21px;
    font-weight: 600;
    color: #0f172a;
    margin-top: 25px;
    margin-bottom: 15px;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}

[data-testid="stSidebar"] h2 {
    color: #0f172a;
}

/* Cards */

.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
    height: 115px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}

.metric-label {
    color: #64748b;
    font-size: 14px;
}

.metric-value {
    color: #0f172a;
    font-size: 27px;
    font-weight: 700;
    margin-top: 8px;
}

/* Generate button */

.stButton > button {
    width: 100%;
    border-radius: 8px;
    height: 45px;
    background-color: #2563eb;
    color: white;
    border: none;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

/* Download button */

.stDownloadButton > button {
    width: 100%;
    border-radius: 8px;
    height: 45px;
}

/* Dataframe */

[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
}

/* Input boxes */

.stSelectbox > div > div,
.stSlider > div {
    color: #0f172a;
}

/* Info box */

.info-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 30px;
    text-align: center;
    margin-top: 20px;
}

.info-title {
    font-size: 20px;
    font-weight: 600;
    color: #0f172a;
}

.info-text {
    color: #64748b;
    margin-top: 8px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "validation" not in st.session_state:
    st.session_state.validation = None


# =========================================================
# PROJECT HEADER
# =========================================================

st.title("Synthetic Data Generator")

st.subheader(
    "From Synthetic Data to Real-World Stress Testing."
)

st.caption(
    "Generate • Stress • Detect • Validate"
)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <h2 style="
            color:#1e293b;
            font-size:20px;
            margin-bottom:20px;
        ">
            Dataset Configuration
        </h2>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # DOMAIN
    # -----------------------------------------------------

    st.markdown(
        '<p style="color:#1e293b; font-weight:600; margin-bottom:5px;">'
        'Domain'
        '</p>',
        unsafe_allow_html=True
    )

    domain = st.selectbox(
        "Domain",
        [
            "Finance",
            "Healthcare",
            "E-Commerce",
            "IoT"
        ],
        label_visibility="collapsed"
    )

    # -----------------------------------------------------
    # DATASET SIZE
    # -----------------------------------------------------

    st.markdown(
        '<p style="color:#1e293b; font-weight:600; margin-top:15px; margin-bottom:5px;">'
        'Dataset Size'
        '</p>',
        unsafe_allow_html=True
    )

    records = st.selectbox(
        "Dataset Size",
        [
            100,
            500,
            1000,
            2500,
            5000,
            10000
        ],
        index=2,
        label_visibility="collapsed"
    )

    # -----------------------------------------------------
    # EDGE CASE RATE
    # -----------------------------------------------------

    st.markdown(
        '<p style="color:#1e293b; font-weight:600; margin-top:15px; margin-bottom:5px;">'
        '⚠️ Edge-Case Rate (%)'
        '</p>',
        unsafe_allow_html=True
    )

    edge_rate = st.slider(
        "Edge-Case Rate",
        0,
        20,
        10,
        label_visibility="collapsed"
    )

    # -----------------------------------------------------
    # DATA RANGE CONTROLS
    # -----------------------------------------------------

    st.markdown(
        '<h3 style="color:#1e293b; margin-top:25px;">'
        '🎛️ Data Range Controls'
        '</h3>',
        unsafe_allow_html=True
    )

    if domain == "Finance":

        st.markdown(
            '<p style="color:#1e293b; font-weight:600;">👤 Age</p>',
            unsafe_allow_html=True
        )

        age_range = st.slider(
            "Age",
            18,
            70,
            (18, 70),
            key="finance_age",
            label_visibility="collapsed"
        )

        st.markdown(
            '<p style="color:#1e293b; font-weight:600;">💰 Income / Salary (₹)</p>',
            unsafe_allow_html=True
        )

        income_range = st.slider(
            "Income",
            15000,
            200000,
            (15000, 200000),
            step=1000,
            key="finance_income",
            label_visibility="collapsed"
        )

        st.markdown(
            '<p style="color:#1e293b; font-weight:600;">💳 Transaction Amount (₹)</p>',
            unsafe_allow_html=True
        )

        transaction_range = st.slider(
            "Transaction Amount",
            100,
            50000,
            (100, 50000),
            step=100,
            key="finance_transaction",
            label_visibility="collapsed"
        )

        data_ranges = {
            "Age": age_range,
            "Income": income_range,
            "Transaction_Amount": transaction_range
        }

    elif domain == "Healthcare":

        st.markdown(
            '<p style="color:#1e293b; font-weight:600;">👤 Patient Age</p>',
            unsafe_allow_html=True
        )

        age_range = st.slider(
            "Age",
            18,
            90,
            (18, 90),
            key="health_age",
            label_visibility="collapsed"
        )

        st.markdown(
            '<p style="color:#1e293b; font-weight:600;">❤️ Heart Rate (BPM)</p>',
            unsafe_allow_html=True
        )

        heart_rate_range = st.slider(
            "Heart Rate",
            40,
            200,
            (60, 100),
            key="health_heart",
            label_visibility="collapsed"
        )

        st.markdown(
            '<p style="color:#1e293b; font-weight:600;">🩸 Glucose Level</p>',
            unsafe_allow_html=True
        )

        glucose_range = st.slider(
            "Glucose",
            50,
            400,
            (70, 140),
            key="health_glucose",
            label_visibility="collapsed"
        )

        st.markdown(
            '<p style="color:#1e293b; font-weight:600;">🌡️ Body Temperature (°C)</p>',
            unsafe_allow_html=True
        )

        temperature_range = st.slider(
            "Temperature",
            35.0,
            42.0,
            (36.0, 37.5),
            step=0.1,
            key="health_temp",
            label_visibility="collapsed"
        )

        data_ranges = {
            "Age": age_range,
            "Heart_Rate": heart_rate_range,
            "Glucose": glucose_range,
            "Temperature": temperature_range
        }

    elif domain == "E-Commerce":

        st.markdown(
            '<p style="color:#1e293b; font-weight:600;">👤 Customer Age</p>',
            unsafe_allow_html=True
        )

        customer_age_range = st.slider(
            "Customer Age",
            18,
            70,
            (18, 70),
            key="commerce_age",
            label_visibility="collapsed"
        )

        st.markdown(
            '<p style="color:#1e293b; font-weight:600;">🏷️ Product Price (₹)</p>',
            unsafe_allow_html=True
        )

        product_price_range = st.slider(
            "Product Price",
            100,
            20000,
            (100, 20000),
            step=100,
            key="commerce_price",
            label_visibility="collapsed"
        )

        st.markdown(
            '<p style="color:#1e293b; font-weight:600;">📦 Quantity</p>',
            unsafe_allow_html=True
        )

        quantity_range = st.slider(
            "Quantity",
            1,
            100,
            (1, 5),
            key="commerce_quantity",
            label_visibility="collapsed"
        )

        st.markdown(
            '<p style="color:#1e293b; font-weight:600;">💰 Order Value (₹)</p>',
            unsafe_allow_html=True
        )

        order_value_range = st.slider(
            "Order Value",
            200,
            500000,
            (200, 50000),
            step=500,
            key="commerce_order",
            label_visibility="collapsed"
        )

        data_ranges = {
            "Customer_Age": customer_age_range,
            "Product_Price": product_price_range,
            "Quantity": quantity_range,
            "Order_Value": order_value_range
        }

    elif domain == "IoT":

        st.markdown(
            '<p style="color:#1e293b; font-weight:600;">🌡️ Temperature (°C)</p>',
            unsafe_allow_html=True
        )

        temperature_range = st.slider(
            "Temperature",
            0.0,
            120.0,
            (20.0, 40.0),
            step=1.0,
            key="iot_temp",
            label_visibility="collapsed"
        )

        st.markdown(
            '<p style="color:#1e293b; font-weight:600;">💧 Humidity (%)</p>',
            unsafe_allow_html=True
        )

        humidity_range = st.slider(
            "Humidity",
            0.0,
            100.0,
            (30.0, 80.0),
            step=1.0,
            key="iot_humidity",
            label_visibility="collapsed"
        )

        st.markdown(
            '<p style="color:#1e293b; font-weight:600;">🌬️ Pressure</p>',
            unsafe_allow_html=True
        )

        pressure_range = st.slider(
            "Pressure",
            900.0,
            1100.0,
            (980.0, 1040.0),
            step=1.0,
            key="iot_pressure",
            label_visibility="collapsed"
        )

        st.markdown(
            '<p style="color:#1e293b; font-weight:600;">📳 Vibration</p>',
            unsafe_allow_html=True
        )

        vibration_range = st.slider(
            "Vibration",
            0.0,
            50.0,
            (0.0, 10.0),
            step=1.0,
            key="iot_vibration",
            label_visibility="collapsed"
        )

        st.markdown(
            '<p style="color:#1e293b; font-weight:600;">⚡ Voltage (V)</p>',
            unsafe_allow_html=True
        )

        voltage_range = st.slider(
            "Voltage",
            150.0,
            300.0,
            (220.0, 240.0),
            step=1.0,
            key="iot_voltage",
            label_visibility="collapsed"
        )

        data_ranges = {
            "Temperature": temperature_range,
            "Humidity": humidity_range,
            "Pressure": pressure_range,
            "Vibration": vibration_range,
            "Voltage": voltage_range
        }

# =========================================================
# SCENARIO LAB
# =========================================================

st.markdown("###  Scenario Lab")

scenario_options = {
    "Finance": [
        "Normal Operations",
        "Fraud Spike",
        "High Value Transactions",
        "Unusual Transaction Burst"
    ],

    "Healthcare": [
        "Normal Patients",
        "Vital Sign Spike",
        "Glucose Risk",
        "Emergency Conditions"
    ],

    "E-Commerce": [
        "Normal Orders",
        "Flash Sale",
        "Bulk Orders",
        "Payment Spike"
    ],

    "IoT": [
        "Normal Sensors",
        "Overheating",
        "Sensor Failure",
        "Abnormal Vibration"
    ]
}

scenario = st.selectbox(
    "Choose a scenario",
    scenario_options[domain]
)
st.write("")

generate = st.button(
        "Generate Dataset"
    )


# =========================================================
# GENERATE DATASET
# =========================================================

if generate:

    with st.spinner("Generating synthetic dataset..."):

        # Generate synthetic data
     df = generate_dataset(
        records=records,
        edge_rate=edge_rate,
        scenario=scenario,
        domain=domain,
        ranges=data_ranges
)

        # Detect anomalies
     df = detect_anomalies(
        df,
        data_ranges=data_ranges
    )

        # Validate dataset
     validation = validate_dataset(
         df,
         ranges=data_ranges
     )

        # Store results
    st.session_state.df = df
    st.session_state.validation = validation


# =========================================================
# GET CURRENT DATA
# =========================================================

df = st.session_state.df
validation = st.session_state.validation


# =========================================================
# DASHBOARD HEADER
# =========================================================

st.markdown(
    '<div class="section-title">Dataset Overview</div>',
    unsafe_allow_html=True
)


# =========================================================
# METRIC CARDS
# =========================================================

c1, c2, c3, c4 = st.columns(4)


if df is not None:

    total_records = len(df)

    edge_cases = (
        df["Edge_Case"] == "Edge Case"
    ).sum()

    ml_anomalies = (
        df["ML_Result"] == "Anomaly"
    ).sum()

    data_quality = validation["Completeness"]

else:

    total_records = "—"
    edge_cases = "—"
    ml_anomalies = "—"
    data_quality = "—"


with c1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Records Generated</div>
            <div class="metric-value">{total_records}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Edge Cases</div>
            <div class="metric-value">{edge_cases}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c3:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">ML Anomalies</div>
            <div class="metric-value">{ml_anomalies}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c4:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Data completeness</div>
            <div class="metric-value">{data_quality}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# DATASET PREVIEW
# =========================================================

st.markdown(
    '<div class="section-title">Dataset Preview</div>',
    unsafe_allow_html=True
)


if df is not None:

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )

else:

    preview_data = {
        "Record_ID": [
            "—",
            "—",
            "—",
            "—",
            "—"
        ],
        "Age": [
            "—",
            "—",
            "—",
            "—",
            "—"
        ],
        "Income": [
            "—",
            "—",
            "—",
            "—",
            "—"
        ],
        "Transaction_Amount": [
            "—",
            "—",
            "—",
            "—",
            "—"
        ],
        "Transaction_Type": [
            "—",
            "—",
            "—",
            "—",
            "—"
        ],
        "Edge_Case": [
            "—",
            "—",
            "—",
            "—",
            "—"
        ]
    }

    st.dataframe(
        preview_data,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# ANALYTICS
# =========================================================

st.markdown(
    '<div class="section-title">Analytics</div>',
    unsafe_allow_html=True
)

if df is not None:

    col1, col2 = st.columns(2)

    # -------------------------
    # Edge Case Pie Chart
    # -------------------------
    edge_counts = df["Edge_Case"].value_counts().reset_index()
    edge_counts.columns = ["Status", "Count"]

    fig_edge = px.pie(
        edge_counts,
        names="Status",
        values="Count",
        title="Edge Case Distribution",
        hole=0.35
    )

    fig_edge.update_layout(
        margin=dict(t=50, b=20, l=20, r=20),
        legend_title_text="Status"
    )

    col1.plotly_chart(
        fig_edge,
        use_container_width=True,
        config={"displayModeBar": False}
    )


    # -------------------------
    # ML Anomaly Pie Chart
    # -------------------------
    anomaly_counts = df[
        "Anomaly_Status"
    ].value_counts().reset_index()

    anomaly_counts.columns = [
        "Result",
        "Count"
    ]

    fig_anomaly = px.pie(
        anomaly_counts,
        names="Result",
        values="Count",
        title="ML Anomaly Detection",
        hole=0.35
    )

    fig_anomaly.update_layout(
        margin=dict(t=50, b=20, l=20, r=20),
        legend_title_text="Result"
    )

    col2.plotly_chart(
        fig_anomaly,
        use_container_width=True,
        config={"displayModeBar": False}
    )
else:
    st.info("Generate a dataset to view analytics.")

# =========================================================
# DATA HEALTH
# =========================================================

st.markdown(
    '<div class="section-title">🧠 Data Health</div>',
    unsafe_allow_html=True
)

if df is not None:

    health = st.session_state.validation

    health_score = health["Health_Score"]
    health_status = health["Health_Status"]

    # -----------------------------------------------------
    # HEALTH SCORE
    # -----------------------------------------------------

    col_score, col_status = st.columns([1, 2])

    with col_score:

        st.metric(
            label="Data Health Score",
            value=f"{health_score}/100"
        )

    with col_status:

        if health_score >= 90:

            st.success(
                f"🟢 {health_status}"
            )

        elif health_score >= 75:

            st.warning(
                f"🟡 {health_status}"
            )

        else:

            st.error(
                f"🔴 {health_status}"
            )

    st.divider()

    # -----------------------------------------------------
    # HEALTH DETAILS
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            f"""
            **✓ Completeness**  
            {health["Completeness"]}%

            **✓ Duplicates**  
            {health["Duplicate_Records"]}

            **✓ Schema**  
            {health["Schema_Status"]}
            """
        )

    with col2:

        st.markdown(
            f"""
            **✓ Valid Ranges**  
            {health["Valid_Ranges"]}%

            **⚠ Anomalies**  
            {health["Anomaly_Percentage"]}%

            **🤖 Anomaly Records**  
            {health["Anomaly_Count"]}
            """
        )

else:

    st.info(
        "Generate a dataset to view Data Health."
    )

# =========================================================
# ANOMALY INVESTIGATION
# =========================================================

st.markdown(
    '<div class="section-title">🚨 Anomaly Investigation</div>',
    unsafe_allow_html=True
)

if df is not None:

    # Get all anomaly records
    anomaly_df = df[
        df["Anomaly_Status"] == "Anomaly"
    ]

    # -----------------------------------------------------
    # NO ANOMALIES
    # -----------------------------------------------------

    if len(anomaly_df) == 0:

        st.success(
            "✅ No anomalies detected in the current dataset."
        )

    # -----------------------------------------------------
    # ANOMALIES FOUND
    # -----------------------------------------------------

    else:

        selected_record = st.selectbox(
            "Select an anomaly record",
            anomaly_df["Record_ID"].tolist(),
            key="anomaly_record_selector"
        )

        # Get selected record
        record = anomaly_df[
            anomaly_df["Record_ID"] == selected_record
        ].iloc[0]

        # -------------------------------------------------
        # ANOMALY SUMMARY
        # -------------------------------------------------

        st.markdown(
            "### 🚨 Anomaly Detected"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown("**Record ID**")
            st.write(record["Record_ID"])

        with col2:

            st.markdown("**Status**")

            if record["Range_Violation"] == "Yes":

                st.error("RANGE VIOLATION")

            else:

                st.warning("ML ANOMALY")

        with col3:

            st.markdown("**Scenario**")
            st.write(record["Scenario"])

        # -------------------------------------------------
        # RECORD DETAILS
        # -------------------------------------------------

        st.markdown("### 📊 Record Details")

        details = {}

        for column in record.index:

            if column not in [
                "Record_ID",
                "ML_Result",
                "Anomaly_Status",
                "Range_Violation",
                "Detected_Signals"
            ]:

                details[column] = record[column]

        detail_df = pd.DataFrame(
            list(details.items()),
            columns=[
                "Feature",
                "Value"
            ]
        )

       




# =========================================================
# DATA VALIDATION
# =========================================================
columns=["Feature", "Value"]


st.dataframe(
            detail_df,
            use_container_width=True,
            hide_index=True
        )


        # =====================================================
        # ML CLASSIFICATION
        # =====================================================

st.markdown("### 🤖 Anomaly Analysis")

if record["Range_Violation"] == "Yes":

    st.error("🚨 RANGE VIOLATION")

else:

    st.warning("🤖 ML ANOMALY")

st.markdown("### 🔎 Detected Signals")

signals = record["Detected_Signals"].split(" | ")

for signal in signals:
    st.success(f"✓ {signal}")

else:

        st.info(
            "No anomalies detected in the current dataset."
        )



st.info(
        "Generate a dataset to investigate anomalies."
    )




# =========================================================
# DATA VALIDATION
# =========================================================

st.markdown(
    '<div class="section-title">Data Validation</div>',
    unsafe_allow_html=True
)


v1, v2, v3 = st.columns(3)


# ---------------------------------------------------------
# COMPLETENESS
# ---------------------------------------------------------

with v1:

    st.write("Completeness")

    if validation is not None:

        completeness = validation["Completeness"]

        st.progress(
            completeness / 100
        )

        st.write(
            f"{completeness}%"
        )

    else:

        st.progress(0)

        st.write("—")


# ---------------------------------------------------------
# DUPLICATES
# ---------------------------------------------------------

with v2:

    st.write("Duplicate Records")

    if validation is not None:

        st.markdown(
            f"""
            <h3 style='color:#0f172a;'>
                {validation["Duplicate_Records"]}
            </h3>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            "<h3 style='color:#0f172a;'>—</h3>",
            unsafe_allow_html=True
        )


# ---------------------------------------------------------
# MISSING VALUES
# ---------------------------------------------------------

with v3:

    st.write("Missing Values")

    if validation is not None:

        st.markdown(
            f"""
            <h3 style='color:#0f172a;'>
                {validation["Missing_Values"]}
            </h3>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            "<h3 style='color:#0f172a;'>—</h3>",
            unsafe_allow_html=True
        )


# =========================================================
# VALIDATION STATUS
# =========================================================

if validation is not None:

    if validation["Validation_Status"] == "PASS":

        st.success(
            "Dataset validation passed successfully."
        )

    else:

        st.warning(
            "Dataset needs further checking."
        )


# =========================================================
# EXPORT
# =========================================================

st.markdown(
    '<div class="section-title">Export</div>',
    unsafe_allow_html=True
)


if df is not None:

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Synthetic Dataset",
        data=csv_data,
        file_name="synthetic_dataset.csv",
        mime="text/csv"
    )

else:

    st.download_button(
        "Download Synthetic Dataset",
        data="",
        file_name="synthetic_dataset.csv",
        mime="text/csv",
        disabled=True
    )


# =========================================================
# EMPTY STATE
# =========================================================

if df is None:

    st.markdown("""
    <div class="info-box">

        <div class="info-title">
            Ready to generate your dataset
        </div>

        <div class="info-text">
            Configure your dataset from the left panel
            and start a new synthetic data generation run.
        </div>

    </div>
    """, unsafe_allow_html=True)