from backend.generator import generate_dataset
from backend.anomaly import detect_anomalies
from backend.validator import validate_dataset


# -----------------------------------------
# STEP 1: Generate dataset
# -----------------------------------------

df = generate_dataset(
    records=1000,
    edge_rate=10,
    scenario="High Risk"
)


# -----------------------------------------
# STEP 2: Detect anomalies
# -----------------------------------------

df = detect_anomalies(df)


# -----------------------------------------
# STEP 3: Validate dataset
# -----------------------------------------

validation = validate_dataset(df)


# -----------------------------------------
# STEP 4: Display validation results
# -----------------------------------------

print("\n========== DATA VALIDATION ==========\n")

print("Missing Values:")
print(validation["Missing_Values"])

print("\nDuplicate Records:")
print(validation["Duplicate_Records"])

print("\nCompleteness:")
print(f'{validation["Completeness"]}%')

print("\nValidation Status:")
print(validation["Validation_Status"])


# -----------------------------------------
# STEP 5: Final result
# -----------------------------------------

print("\n====================================")

if validation["Validation_Status"] == "PASS":
    print("Dataset validation successful! ✅")
else:
    print("Dataset needs further checking. ⚠️")

print("====================================")