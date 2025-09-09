import pandas as pd
import numpy as np
import streamlit as st
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, mean_absolute_error
import plotly.graph_objects as go

st.set_page_config(page_title="Baku Podium Predictor", layout="centered")
st.title("🏁 Baku GP Podium Predictor")

# ----------------- Load -----------------
@st.cache_data
def load_data(path="data/F1_RaceResult_DriverFeatures.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=["Driver_AvgFinish_Last5",
                           "Driver_PodiumRate_Last5",
                           "Driver_PrevFinish"])
    return df

df = load_data()

# ----------------- Prepare -----------------
X = df[["Driver_AvgFinish_Last5", "Driver_PodiumRate_Last5", "Driver_PrevFinish"]]
y = df["Podium"]

# custom split: Test = 2025 Italy & Netherlands
is_2025 = df["Year"] == 2025
is_test = is_2025 & df["Circuit Name"].isin(["Italy", "Netherlands"])
is_train = ~is_test

X_train, y_train = X[is_train], y[is_train]
X_test,  y_test  = X[is_test],  y[is_test]

# ----------------- Model -----------------
pos_w = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
model = XGBClassifier(
    n_estimators=125,
    learning_rate=0.02,
    max_depth=3,
    subsample=0.6,
    colsample_bytree=0.5,
    random_state=42,
    eval_metric="logloss",
    scale_pos_weight=pos_w
)
model.fit(X_train, y_train)

# ----------------- Evaluate -----------------
y_pred_test = model.predict(X_test)
y_prob_test = model.predict_proba(X_test)[:, 1]
mae_test = mean_absolute_error(y_test, y_prob_test)
# ----------------- Predict Top 3 (Baku) -----------------
from PIL import Image
import plotly.graph_objects as go

latest_2025 = df[is_2025].groupby("Driver Name", as_index=False).tail(1).copy()
X_baku = latest_2025[["Driver_AvgFinish_Last5",
                      "Driver_PodiumRate_Last5",
                      "Driver_PrevFinish"]]
latest_2025["Podium_Prob"] = model.predict_proba(X_baku)[:, 1]

# attach team if missing
if "Team" not in latest_2025.columns:
    team_map = df[is_2025][["Driver Name", "Team"]].drop_duplicates().set_index("Driver Name")["Team"].to_dict()
    latest_2025["Team"] = latest_2025["Driver Name"].map(team_map)

top3 = (latest_2025.sort_values("Podium_Prob", ascending=False)
        .head(3)[["Driver Name", "Team", "Podium_Prob"]]
        .rename(columns={"Driver Name": "Driver", "Podium_Prob": "Prob"}))

# pad if < 3 drivers
while len(top3) < 3:
    top3 = pd.concat([top3, pd.DataFrame([{"Driver": "TBD", "Team": "", "Prob": 0.0}])],
                     ignore_index=True)

# reorder visually: [2nd, 1st, 3rd]
vis = top3.iloc[[1, 0, 2]].reset_index(drop=True)

# ---------- Game-style podium viz ----------
heights = [1.0, 1.4, 0.8]   # block heights
xpos    = [0,   1,   2]     # 2nd, 1st, 3rd order
colors  = ["orange", "orange", "red"]

# load only the logos you have locally
team_logos = {
    "McLaren": Image.open("mclaren.jpg"),
    "Red Bull Racing": Image.open("redbull.jpg"),
    # add more teams later if you want
}

fig = go.Figure()

# blocks
for i in range(3):
    fig.add_shape(
        type="rect",
        x0=xpos[i]-0.45, x1=xpos[i]+0.45,
        y0=0, y1=heights[i],
        line=dict(width=0),
        fillcolor=colors[i]
    )

# rank numbers inside blocks
for i, rank in enumerate([2, 1, 3]):
    fig.add_annotation(
        x=xpos[i], y=heights[i]*0.55,
        text=f"<b>{rank}</b>", showarrow=False,
        font=dict(size=34, color="white")
    )

# team logos just below the rank numbers
for i, row in vis.iterrows():
    team = row["Team"]
    logo = team_logos.get(team)
    if logo:
        fig.add_layout_image(
            dict(
                source=logo,                   # PIL image object
                x=xpos[i], y=heights[i]*0.25, # position inside block
                xref="x", yref="y",
                sizex=0.5, sizey=0.5,
                xanchor="center", yanchor="middle",
                layer="above"
            )
        )

# driver names, teams, probs above blocks
for i, row in vis.iterrows():
    driver = row["Driver"]
    team   = row["Team"]
    prob   = row["Prob"]
    fig.add_annotation(
        x=xpos[i], y=heights[i]+0.05,
        text=f"<b>{driver}</b><br><span style='font-size:12px'>{team}</span><br>"
             f"<span style='font-size:11px'>Prob: {prob:.3f}</span>",
        showarrow=False, yanchor="bottom",
        font=dict(color="black")
    )

fig.update_xaxes(visible=False, range=[-0.8, 2.8])
fig.update_yaxes(visible=False, range=[0, 2.0])
fig.update_layout(
    width=720, height=460,
    margin=dict(l=20, r=20, t=20, b=30),
    plot_bgcolor="white", paper_bgcolor="white"
)

st.subheader("Predicted Podium for Azerbaijan GP(Baku) — 2025")
st.plotly_chart(fig, use_container_width=True)
st.metric("MAE (probabilities vs labels)", f"{mae_test:.4f}")