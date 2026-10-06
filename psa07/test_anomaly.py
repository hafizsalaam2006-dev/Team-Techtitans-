from backend.generator import generate_dataset
from backend.anomaly import detect_anomalies


# --------------------------------------------------
# STEP 1: Generate synthetic dataset
# --------------------------------------------------

df = generate_dataset(
    records=1000,
    edge_rate=10,
    scenario="High Risk"
)


# --------------------------------------------------
# STEP 2: Detect anomalies
# --------------------------------------------------

df = detect_anomalies(df)


# --------------------------------------------------
# STEP 3: Display sample records
# --------------------------------------------------

print("\n========== SAMPLE DATA ==========\n")

print(df.head(10))


# --------------------------------------------------
# STEP 4: Total records
# --------------------------------------------------

print("\n========== DATASET INFORMATION ==========\n")

print("Total Records:")
print(len(df))


# --------------------------------------------------
# STEP 5: Count edge cases
# --------------------------------------------------

edge_cases = (
    df["Edge_Case"] == "Edge Case"
).sum()

print("\nEdge Cases:")
print(edge_cases)


# --------------------------------------------------
# STEP 6: Count ML anomalies
# --------------------------------------------------

anomalies = (
    df["ML_Result"] == "Anomaly"
).sum()

print("\nML Anomalies:")
print(anomalies)


# --------------------------------------------------
# STEP 7: Count normal records
# --------------------------------------------------

normal = (
    df["ML_Result"] == "Normal"
).sum()

print("\nNormal Records:")
print(normal)


# --------------------------------------------------
# STEP 8: Show anomaly distribution
# --------------------------------------------------

print("\n========== ML RESULT ==========\n")

print(
    df["ML_Result"].value_counts()
)


# --------------------------------------------------
# STEP 9: Show detected anomalies
# --------------------------------------------------

print("\n========== DETECTED ANOMALIES ==========\n")

print(
    df[
        df["ML_Result"] == "Anomaly"
    ].head(10)
)