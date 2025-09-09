# 🏎️ F1 Podium Predictor

Predict podium finishes for Formula 1 races using machine learning (XGBoost) and visualize results in a **game-style podium view** built with Streamlit + Plotly.

---

## 📌 Project Overview
This project uses historical F1 race data (2019–2025) to:
- Engineer **driver form features** (average finish, podium rate, last finish)  
- Add **team-level rolling averages**  
- Train an **XGBoost model** to predict podium finishes  
- Hold out **2025 Italy & Netherlands** as the test set  
- Visualize podium predictions for the next race (Baku GP) in a **game-style podium** with driver names, teams, probabilities, and logos  

---

## ⚙️ Tech Stack
- **Python**: pandas, numpy, scikit-learn, xgboost  
- **Streamlit**: interactive app for predictions and visualization  
- **Plotly**: podium graphic (blocks, ranks, annotations)  

---
## 📊 Features
- ✅ **Train/test split** → 2025 Italy & Netherlands are used as a hold-out test set  
- ✅ **Evaluation metrics** → Classification Report + MAE on test set  
- ✅ **Podium prediction** → Top 3 drivers for the next race based on most recent form  
- ✅ **Game-style podium view** →  
  - Rank numbers (1, 2, 3) inside podium blocks  
  - Driver name, team, and probability shown above each block  

---

## 🖼️ Prediction


<img width="704" height="460" alt="newplot" src="https://github.com/user-attachments/assets/a4a7d78e-4227-4010-9df4-e34b2135ee9d" />


---
## PS
- I also used Gradient Boosting Classifer as my baseline model and also tried to fine tune it with the help of XGBoost Classifer using race data from the start of 2019. 

---

## ✨ Next Steps
- Try to integrate using FASTF1 API
- Extend dataset with **qualifying results, pit stops, weather conditions** for improved accuracy  

---

## 👤 Author
**Sai Krishna**  

