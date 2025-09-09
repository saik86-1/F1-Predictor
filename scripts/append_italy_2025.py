import pandas as pd
from pathlib import Path

CSV_PATH = Path("F1_RaceResult_2019_2025.csv")
BACKUP_PATH = CSV_PATH.with_suffix(".bak.csv")

# Backup once (skip if it already exists)
if not BACKUP_PATH.exists():
    BACKUP_PATH.write_bytes(CSV_PATH.read_bytes())

rows = [
    # Year, Circuit Name, Driver Number, Driver Name, Team, Starting Grid, Final Position
    (2025, "Italy", 1,  "Max Verstappen",  "Red Bull Racing", 1,  "1"),
    (2025, "Italy", 4,  "Lando Norris",    "McLaren",         2,  "2"),
    (2025, "Italy", 81, "Oscar Piastri",   "McLaren",         3,  "3"),
    (2025, "Italy", 16, "Charles Leclerc", "Ferrari",         4,  "4"),
    (2025, "Italy", 63, "George Russell",  "Mercedes",        5,  "5"),
    (2025, "Italy", 44, "Lewis Hamilton",  "Ferrari",         10, "6"),
    (2025, "Italy", 23, "Alexander Albon", "Williams",        14, "7"),
    (2025, "Italy", 5,  "Gabriel Bortoleto","Kick Sauber",    7,  "8"),
    (2025, "Italy", 12, "Kimi Antonelli",  "Mercedes",        6,  "9"),
    (2025, "Italy", 6,  "Isack Hadjar",    "Racing Bulls",    "Pit Lane", "10"),
    (2025, "Italy", 55, "Carlos Sainz",    "Williams",        13, "11"),
    (2025, "Italy", 87, "Oliver Bearman",  "Haas",            11, "12"),
    (2025, "Italy", 22, "Yuki Tsunoda",    "Red Bull Racing", 9,  "13"),
    (2025, "Italy", 30, "Liam Lawson",     "Racing Bulls",    18, "14"),
    (2025, "Italy", 31, "Esteban Ocon",    "Haas",            15, "15"),
    (2025, "Italy", 10, "Pierre Gasly",    "Alpine",          "Pit Lane", "16"),
    (2025, "Italy", 43, "Franco Colapinto","Alpine",          17, "17"),
    (2025, "Italy", 18, "Lance Stroll",    "Aston Martin",    16, "18"),
    (2025, "Italy", 14, "Fernando Alonso", "Aston Martin",    8,  "DNF"),
    (2025, "Italy", 27, "Nico Hülkenberg", "Kick Sauber",     12, "DNS"),
]

new_df = pd.DataFrame(rows, columns=[
    "Year","Circuit Name","Driver Number","Driver Name","Team","Starting Grid","Final Position"
])

# enforce dtypes you specified: ints for Year & Driver Number; others object
new_df["Year"] = new_df["Year"].astype("int64")
new_df["Driver Number"] = new_df["Driver Number"].astype("int64")

# read, append, and save
df = pd.read_csv(CSV_PATH)
df_out = pd.concat([df, new_df], ignore_index=True)
df_out.to_csv(CSV_PATH, index=False)

print("Appended", len(new_df), "rows for Italy 2025. New total rows:", len(df_out))
