import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df, data_ranges=None):

    result = df.copy()

    # =========================================================
    # 1. RANGE VIOLATION DETECTION
    # =========================================================

    result["Range_Violation"] = "No"

    range_signals = {}

    if data_ranges:

        for column, selected_range in data_ranges.items():

            if column not in result.columns:
                continue

            min_value, max_value = selected_range

            violation = (
                (result[column] < min_value) |
                (result[column] > max_value)
            )

            result.loc[
                violation,
                "Range_Violation"
            ] = "Yes"

            range_signals[column] = violation

    # =========================================================
    # 2. ML ANOMALY DETECTION
    # =========================================================

    numeric_columns = result.select_dtypes(
        include=np.number
    ).columns.tolist()

    if "Record_ID" in numeric_columns:
        numeric_columns.remove("Record_ID")

    # Remove helper columns if present
    numeric_columns = [
        col for col in numeric_columns
        if col not in [
            "Range_Violation"
        ]
    ]

    # Default
    result["ML_Result"] = "Normal"

    if len(numeric_columns) > 0:

        features = result[numeric_columns].copy()

        # Use a LOW contamination value.
        # ML is supporting the rule-based detection,
        # not forcing every dataset to have 5% anomalies.
        model = IsolationForest(
            n_estimators=100,
            contamination=0.02,
            random_state=42
        )

        predictions = model.fit_predict(features)

        result["ML_Result"] = np.where(
            predictions == -1,
            "Anomaly",
            "Normal"
        )

    # =========================================================
    # 3. FINAL ANOMALY STATUS
    # =========================================================

    result["Anomaly_Status"] = np.where(
        (result["Range_Violation"] == "Yes") |
        (result["ML_Result"] == "Anomaly"),
        "Anomaly",
        "Normal"
    )

    # =========================================================
    # 4. DETECTED SIGNALS
    # =========================================================

    def find_signals(row):

        signals = []

        # -----------------------------------------------------
        # RANGE VIOLATIONS
        # -----------------------------------------------------

        if data_ranges:

            for column, selected_range in data_ranges.items():

                if column not in row.index:
                    continue

                min_value, max_value = selected_range
                value = row[column]

                if value < min_value or value > max_value:

                    signals.append(
                        f"{column.replace('_', ' ')} "
                        f"outside selected range "
                        f"({min_value:g} - {max_value:g})"
                    )

        # -----------------------------------------------------
        # FINANCE
        # -----------------------------------------------------

        if "Transaction_Amount" in row.index:

            if row["Transaction_Amount"] > 50000:
                signals.append(
                    "Extreme transaction amount"
                )

        # -----------------------------------------------------
        # HEALTHCARE
        # -----------------------------------------------------

        if "Heart_Rate" in row.index:

            if row["Heart_Rate"] > 120:
                signals.append(
                    "Unusual heart rate"
                )

        if "Glucose" in row.index:

            if row["Glucose"] > 180:
                signals.append(
                    "Unusual glucose level"
                )

        if "Temperature" in row.index:

            # Healthcare temperature
            if (
                "Heart_Rate" in row.index
                and row["Temperature"] > 39
            ):
                signals.append(
                    "High body temperature"
                )

            # IoT temperature
            elif (
                "Device_Type" in row.index
                and row["Temperature"] > 70
            ):
                signals.append(
                    "Extreme sensor temperature"
                )

        # -----------------------------------------------------
        # E-COMMERCE
        # -----------------------------------------------------

        if "Order_Value" in row.index:

            if row["Order_Value"] > 50000:
                signals.append(
                    "Unusually high order value"
                )

        if "Quantity" in row.index:

            if row["Quantity"] > 20:
                signals.append(
                    "Unusually large quantity"
                )

        # -----------------------------------------------------
        # IOT
        # -----------------------------------------------------

        if "Vibration" in row.index:

            if row["Vibration"] > 20:
                signals.append(
                    "Abnormal vibration"
                )

        if "Voltage" in row.index:

            if (
                row["Voltage"] > 260
                or row["Voltage"] < 180
            ):
                signals.append(
                    "Abnormal voltage"
                )

        # -----------------------------------------------------
        # EDGE CASE
        # -----------------------------------------------------

        if row.get("Edge_Case", "Normal") == "Edge Case":

            signals.append(
                "Stress-case record"
            )

        # -----------------------------------------------------
        # ML SIGNAL
        # -----------------------------------------------------

        if row["ML_Result"] == "Anomaly":

            signals.append(
                "Unusual multivariate pattern"
            )

        # -----------------------------------------------------
        # FALLBACK
        # -----------------------------------------------------

        if not signals:

            signals.append(
                "No significant anomaly signal"
            )

        return " | ".join(signals)

    result["Detected_Signals"] = result.apply(
        find_signals,
        axis=1
    )

    return result