# BlitzWatch 🚨

**BlitzWatch** is an AI-powered tool that predicts whether a defensive blitz is likely to occur on an NFL passing play using public play-by-play data. Built with data from `nflfastR`, this project leverages machine learning to identify pressure tendencies based on down, distance, score, and game context.

---

## 📅 Project Goals
- Predict blitz/no-blitz using real NFL data
- Analyze key indicators of defensive pressure
- Visualize model predictions and feature importance
- Compare CPU vs CUDA performance (optional later)

---

## 🔧 Tech Stack
- Python
- pandas, matplotlib, seaborn
- LightGBM (for fast modeling)
- SHAP (for interpretability)
- nfl_data_py (`nflfastR` wrapper)

---

## 📚 Data Source
- [nflfastR](https://www.nflfastr.com/)
- Seasons: 2018 to 2023
- Filtered to passing plays only
- Label approximated via `qb_hit` or `pressure`

---

## 🎯 Features Used
- Down & distance
- Field position (yardline)
- Quarter
- Shotgun formation
- No-huddle
- Score differential
- Time remaining
- Pass length/location

---

## 🎓 How It Works
1. Load play-by-play data using `nfl_data_py`
2. Engineer features and label `blitz` plays
3. Train a LightGBM binary classifier
4. Analyze accuracy, recall, precision
5. Visualize important features using SHAP

---

## 📊 Sample Output
```
Prediction: Blitz (87.4% confidence)
Top features: Down = 3, Distance = 8, Shotgun = True, Score Diff = -10
```

---

## 🚀 Future Plans
- Integrate Big Data Bowl tracking data for true blitz labels
- Train CNN/RNN on player movement sequences
- Deploy via Streamlit or Flask
- CUDA-accelerated deep model for faster inference

---

## 🚀 Get Started
```bash
pip install nfl_data_py lightgbm pandas matplotlib shap seaborn
```
Then run the notebook or script from `/src`.

---

## 🧡 Author
**Keenan Jajeh**  
[github.com/keenan34](https://github.com/keenan34)
