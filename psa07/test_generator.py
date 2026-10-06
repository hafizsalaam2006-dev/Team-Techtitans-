from backend.generator import generate_dataset


df = generate_dataset(
    records=1000,
    edge_rate=10,
    scenario="High Risk"
)

print(df.head())

print("\nTotal Records:")
print(len(df))

print("\nEdge Cases:")
print(
    (df["Edge_Case"] == "Edge Case").sum()
)