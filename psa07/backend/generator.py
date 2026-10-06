import pandas as pd
import numpy as np


def generate_dataset(
    records=1000,
    edge_rate=10,
    scenario="Normal Operations",
    domain="Finance",
    ranges=None
):

    rng = np.random.default_rng()

    if ranges is None:
        ranges = {}

    # =========================================================
    # FINANCE
    # =========================================================

    if domain == "Finance":

        age_min, age_max = ranges.get("Age", (18, 70))
        income_min, income_max = ranges.get(
            "Income", (15000, 200000)
        )
        transaction_min, transaction_max = ranges.get(
            "Transaction_Amount", (100, 50000)
        )

        df = pd.DataFrame({
            "Record_ID": [
                f"REC-{i:05d}"
                for i in range(1, records + 1)
            ],

            "Age": rng.integers(
                age_min,
                age_max + 1,
                records
            ),

            "Income": rng.integers(
                income_min,
                income_max + 1,
                records
            ),

            "Transaction_Amount": np.round(
                rng.uniform(
                    transaction_min,
                    transaction_max,
                    records
                ),
                2
            ),

            "Transaction_Type": rng.choice(
                [
                    "UPI",
                    "Card",
                    "Bank Transfer",
                    "Wallet"
                ],
                records
            ),

            "Location": rng.choice(
                [
                    "Chennai",
                    "Coimbatore",
                    "Bengaluru",
                    "Mumbai",
                    "Delhi"
                ],
                records
            )
        })

        # Scenario Lab

        if scenario == "Fraud Spike":

            count = int(records * 0.15)

            indices = rng.choice(
                records,
                count,
                replace=False
            )

            df.loc[
                indices,
                "Transaction_Amount"
            ] = rng.integers(
                100000,
                1000000,
                count
            )

            df.loc[
                indices,
                "Transaction_Type"
            ] = "Card"

        elif scenario == "High Value Transactions":

            count = int(records * 0.20)

            indices = rng.choice(
                records,
                count,
                replace=False
            )

            df.loc[
                indices,
                "Transaction_Amount"
            ] = rng.integers(
                50000,
                500000,
                count
            )

        elif scenario == "Unusual Transaction Burst":

            count = int(records * 0.25)

            indices = rng.choice(
                records,
                count,
                replace=False
            )

            df.loc[
                indices,
                "Transaction_Amount"
            ] *= rng.integers(
                5,
                15,
                count
            )

    # =========================================================
    # HEALTHCARE
    # =========================================================

    elif domain == "Healthcare":

        age_min, age_max = ranges.get(
            "Age",
            (18, 90)
        )

        heart_min, heart_max = ranges.get(
            "Heart_Rate",
            (60, 100)
        )

        glucose_min, glucose_max = ranges.get(
            "Glucose",
            (70, 140)
        )

        temp_min, temp_max = ranges.get(
            "Temperature",
            (36.0, 37.5)
        )

        df = pd.DataFrame({

            "Record_ID": [
                f"REC-{i:05d}"
                for i in range(1, records + 1)
            ],

            "Age": rng.integers(
                age_min,
                age_max + 1,
                records
            ),

            "Heart_Rate": rng.integers(
                heart_min,
                heart_max + 1,
                records
            ),

            "Blood_Pressure": rng.integers(
                90,
                141,
                records
            ),

            "Glucose": rng.integers(
                glucose_min,
                glucose_max + 1,
                records
            ),

            "Temperature": np.round(
                rng.uniform(
                    temp_min,
                    temp_max,
                    records
                ),
                1
            ),

            "Gender": rng.choice(
                [
                    "Male",
                    "Female"
                ],
                records
            )
        })

        # Scenario Lab

        if scenario == "Vital Sign Spike":

            count = int(records * 0.15)

            indices = rng.choice(
                records,
                count,
                replace=False
            )

            df.loc[
                indices,
                "Heart_Rate"
            ] = rng.integers(
                130,
                190,
                count
            )

            df.loc[
                indices,
                "Blood_Pressure"
            ] = rng.integers(
                160,
                220,
                count
            )

        elif scenario == "Glucose Risk":

            count = int(records * 0.20)

            indices = rng.choice(
                records,
                count,
                replace=False
            )

            df.loc[
                indices,
                "Glucose"
            ] = rng.integers(
                200,
                400,
                count
            )

        elif scenario == "Emergency Conditions":

            count = int(records * 0.20)

            indices = rng.choice(
                records,
                count,
                replace=False
            )

            df.loc[
                indices,
                "Heart_Rate"
            ] = rng.integers(
                140,
                200,
                count
            )

            df.loc[
                indices,
                "Glucose"
            ] = rng.integers(
                250,
                450,
                count
            )

            df.loc[
                indices,
                "Temperature"
            ] = np.round(
                rng.uniform(
                    39.0,
                    41.5,
                    count
                ),
                1
            )

    # =========================================================
    # E-COMMERCE
    # =========================================================

    elif domain == "E-Commerce":

        age_min, age_max = ranges.get(
            "Customer_Age",
            (18, 70)
        )

        price_min, price_max = ranges.get(
            "Product_Price",
            (100, 20000)
        )

        quantity_min, quantity_max = ranges.get(
            "Quantity",
            (1, 5)
        )

        order_min, order_max = ranges.get(
            "Order_Value",
            (200, 50000)
        )

        df = pd.DataFrame({

            "Record_ID": [
                f"REC-{i:05d}"
                for i in range(1, records + 1)
            ],

            "Customer_Age": rng.integers(
                age_min,
                age_max + 1,
                records
            ),

            "Product_Price": rng.integers(
                price_min,
                price_max + 1,
                records
            ),

            "Quantity": rng.integers(
                quantity_min,
                quantity_max + 1,
                records
            ),

            "Order_Value": rng.integers(
                order_min,
                order_max + 1,
                records
            ),

            "Product_Category": rng.choice(
                [
                    "Electronics",
                    "Fashion",
                    "Home",
                    "Beauty",
                    "Grocery"
                ],
                records
            ),

            "Payment_Method": rng.choice(
                [
                    "UPI",
                    "Card",
                    "COD",
                    "Wallet"
                ],
                records
            )
        })

        # Scenario Lab

        if scenario == "Flash Sale":

            count = int(records * 0.30)

            indices = rng.choice(
                records,
                count,
                replace=False
            )

            df.loc[
                indices,
                "Product_Price"
            ] = rng.integers(
                500,
                10000,
                count
            )

            df.loc[
                indices,
                "Quantity"
            ] = rng.integers(
                3,
                15,
                count
            )

            df.loc[
                indices,
                "Order_Value"
            ] = (
                df.loc[
                    indices,
                    "Product_Price"
                ]
                *
                df.loc[
                    indices,
                    "Quantity"
                ]
            )

        elif scenario == "Bulk Orders":

            count = int(records * 0.20)

            indices = rng.choice(
                records,
                count,
                replace=False
            )

            df.loc[
                indices,
                "Quantity"
            ] = rng.integers(
                10,
                100,
                count
            )

            df.loc[
                indices,
                "Order_Value"
            ] = (
                df.loc[
                    indices,
                    "Product_Price"
                ]
                *
                df.loc[
                    indices,
                    "Quantity"
                ]
            )

        elif scenario == "Payment Spike":

            count = int(records * 0.25)

            indices = rng.choice(
                records,
                count,
                replace=False
            )

            df.loc[
                indices,
                "Order_Value"
            ] = rng.integers(
                50000,
                500000,
                count
            )

    # =========================================================
    # IOT
    # =========================================================

    elif domain == "IoT":

        temp_min, temp_max = ranges.get(
            "Temperature",
            (20.0, 40.0)
        )

        humidity_min, humidity_max = ranges.get(
            "Humidity",
            (30.0, 80.0)
        )

        pressure_min, pressure_max = ranges.get(
            "Pressure",
            (980.0, 1040.0)
        )

        vibration_min, vibration_max = ranges.get(
            "Vibration",
            (0.0, 10.0)
        )

        voltage_min, voltage_max = ranges.get(
            "Voltage",
            (220.0, 240.0)
        )

        df = pd.DataFrame({

            "Record_ID": [
                f"REC-{i:05d}"
                for i in range(1, records + 1)
            ],

            "Temperature": np.round(
                rng.uniform(
                    temp_min,
                    temp_max,
                    records
                ),
                2
            ),

            "Humidity": np.round(
                rng.uniform(
                    humidity_min,
                    humidity_max,
                    records
                ),
                2
            ),

            "Pressure": np.round(
                rng.uniform(
                    pressure_min,
                    pressure_max,
                    records
                ),
                2
            ),

            "Vibration": np.round(
                rng.uniform(
                    vibration_min,
                    vibration_max,
                    records
                ),
                2
            ),

            "Voltage": np.round(
                rng.uniform(
                    voltage_min,
                    voltage_max,
                    records
                ),
                2
            ),

            "Device_Type": rng.choice(
                [
                    "Motor",
                    "Pump",
                    "Generator",
                    "Fan"
                ],
                records
            )
        })

        # Scenario Lab

        if scenario == "Overheating":

            count = int(records * 0.20)

            indices = rng.choice(
                records,
                count,
                replace=False
            )

            df.loc[
                indices,
                "Temperature"
            ] = np.round(
                rng.uniform(
                    80,
                    120,
                    count
                ),
                2
            )

        elif scenario == "Sensor Failure":

            count = int(records * 0.10)

            indices = rng.choice(
                records,
                count,
                replace=False
            )

            df.loc[
                indices,
                "Temperature"
            ] = 0

            df.loc[
                indices,
                "Humidity"
            ] = 0

            df.loc[
                indices,
                "Voltage"
            ] = 0

        elif scenario == "Abnormal Vibration":

            count = int(records * 0.20)

            indices = rng.choice(
                records,
                count,
                replace=False
            )

            df.loc[
                indices,
                "Vibration"
            ] = np.round(
                rng.uniform(
                    20,
                    50,
                    count
                ),
                2
            )

    # =========================================================
    # EDGE CASE GENERATION
    # =========================================================

    normal_scenarios = {
        "Normal Operations",
        "Normal Patients",
        "Normal Orders",
        "Normal Sensors"
    }

    actual_edge_rate = edge_rate

    if scenario not in normal_scenarios:
        actual_edge_rate = max(
            edge_rate,
            10
        )

    edge_count = int(
        records * actual_edge_rate / 100
    )

    df["Edge_Case"] = "Normal"

    if edge_count > 0:

        edge_indices = rng.choice(
            records,
            edge_count,
            replace=False
        )

        df.loc[
            edge_indices,
            "Edge_Case"
        ] = "Edge Case"

    # Store scenario

    df["Scenario"] = scenario

    return df