import pandas as pd


def validate_dataset(df, ranges=None):

    if ranges is None:
        ranges = {}

    # =========================================================
    # COMPLETENESS
    # =========================================================

    missing_values = int(
        df.isnull().sum().sum()
    )

    total_cells = (
        df.shape[0] *
        df.shape[1]
    )

    if total_cells > 0:
        completeness = (
            (total_cells - missing_values)
            / total_cells
        ) * 100
    else:
        completeness = 0

    completeness = round(
        completeness,
        2
    )

    # =========================================================
    # DUPLICATES
    # =========================================================

    duplicate_records = int(
        df.duplicated().sum()
    )

    if len(df) > 0:

        duplicate_quality = (
            (len(df) - duplicate_records)
            / len(df)
        ) * 100

    else:

        duplicate_quality = 100

    duplicate_quality = round(
        duplicate_quality,
        2
    )

    # =========================================================
    # SCHEMA
    # =========================================================

    required_columns = [
        "Record_ID",
        "Edge_Case",
        "Scenario",
        "ML_Result",
        "Anomaly_Status"
    ]

    missing_schema_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if len(missing_schema_columns) == 0:

        schema_status = "PASS"
        schema_score = 100

    else:

        schema_status = "CHECK"
        schema_score = 0

    # =========================================================
    # VALID RANGE %
    # =========================================================

    if len(df) > 0 and len(ranges) > 0:

        total_range_checks = 0
        valid_range_checks = 0

        for column, selected_range in ranges.items():

            if column not in df.columns:
                continue

            min_value, max_value = selected_range

            total_range_checks += len(df)

            valid_range_checks += int(
                (
                    (df[column] >= min_value)
                    &
                    (df[column] <= max_value)
                ).sum()
            )

        if total_range_checks > 0:

            valid_ranges = (
                valid_range_checks
                / total_range_checks
            ) * 100

        else:

            valid_ranges = 100

    else:

        valid_ranges = 100

    valid_ranges = round(
        valid_ranges,
        2
    )

    # =========================================================
    # ANOMALY %
    # =========================================================

    if "Anomaly_Status" in df.columns:

        anomaly_count = int(
            (
                df["Anomaly_Status"]
                == "Anomaly"
            ).sum()
        )

        if len(df) > 0:

            anomaly_percentage = (
                anomaly_count /
                len(df)
            ) * 100

        else:

            anomaly_percentage = 0

    else:

        anomaly_count = 0
        anomaly_percentage = 0

    anomaly_percentage = round(
        anomaly_percentage,
        2
    )

    # Percentage of records WITHOUT anomalies
    anomaly_free_percentage = round(
        100 - anomaly_percentage,
        2
    )

    # =========================================================
    # DATA HEALTH SCORE
    # =========================================================

    health_score = (
        completeness * 0.25
        + duplicate_quality * 0.15
        + schema_score * 0.15
        + valid_ranges * 0.25
        + anomaly_free_percentage * 0.20
    )

    health_score = round(
        health_score
    )

    # =========================================================
    # HEALTH STATUS
    # =========================================================

    if health_score >= 90:

        health_status = "HEALTHY"

    elif health_score >= 75:

        health_status = "GOOD"

    elif health_score >= 60:

        health_status = "NEEDS REVIEW"

    else:

        health_status = "POOR"

    return {

        "Missing_Values":
            missing_values,

        "Duplicate_Records":
            duplicate_records,

        "Completeness":
            completeness,

        "Schema_Status":
            schema_status,

        "Valid_Ranges":
            valid_ranges,

        "Anomaly_Count":
            anomaly_count,

        "Anomaly_Percentage":
            anomaly_percentage,

        "Health_Score":
            health_score,

        "Health_Status":
            health_status,

        "Validation_Status":
            (
                "PASS"
                if (
                    missing_values == 0
                    and duplicate_records == 0
                    and schema_status == "PASS"
                )
                else "CHECK"
            )
    }