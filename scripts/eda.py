import pandas as pd

df = pd.read_csv("data/F1_RaceResult_DriverFeatures.csv")

print("Shape:", df.shape)
#print("\nColumn types:\n", df.dtypes)
#print("\nNull values:\n", df.isnull().sum())
print("\nSample rows:\n", df[["Driver Name","Circuit Name","Driver_PrevFinish"]].tail(20))

# unique values check
#for col in ["Year", "Circuit Name", "Team"]:
 #   print(f"\nUnique in {col}: {df[col].nunique()}")
  #  print(df[col].unique()[:10])  # show first 10 unique values
