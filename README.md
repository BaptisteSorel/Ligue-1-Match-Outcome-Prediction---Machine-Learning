# Ligue-1-Match-Outcome-Prediction---Machine-Learning

A machine learning project to predict the outcome of Ligue 1 football matches (Home win / Draw / Away win), built from scratch as part of a quantitative finance curriculum. The project draws explicit parallels with financial markets : predicting match probabilities is structurally similar to pricing derivatives, and comparing model outputs to bookmaker odds mirrors the search for alpha in quantitative trading.

---

## Motivation

Football prediction sits at the intersection of sports analytics and quantitative finance. Bookmaker odds are a form of market pricing — they aggregate vast amounts of information and are generally efficient. The question this project attempts to answer is :

> *Can a data-driven model, built from publicly available statistics, produce probability estimates that are competitive with bookmaker implied probabilities ?*

This project was also an opportunity to develop practical machine learning skills — feature engineering, model selection, calibration analysis — applied to a real, noisy, and inherently hard-to-predict dataset.

---

## Project structure
```
ML-Ligue1/
│
├── data/
│   ├── raw/
│   │   ├── understat_ligue1_matches.xlsx    # Raw match data (Understat)
│   │   └── transfermarkt_values.xlsx        # Squad market values (Transfermarkt)
│   ├── ligue1_features.csv                  # Engineered feature dataset
│   └── Odds_24_25.csv                       # Bet365 odds for 2024/25 season
│
├── notebooks/
│   ├── 01_data_exploration.ipynb            # EDA — distributions, correlations, home advantage
│   ├── 02_feature_engineering.ipynb         # Feature construction pipeline
│   ├── 03_modelisation.ipynb                # Model training and evaluation
│   └── 04_evaluation_improvement.ipynb      # Bookmaker comparison, prediction examples
│
├── models/
│   └── model_rf.pkl                         # Saved Random Forest model
│
└── README.md
```

---

## Data sources

| Source | Content | Seasons |
|--------|---------|---------|
| [Understat](https://understat.com) | Match results, xG, shots, PPDA, deep passes | 2016/17 → 2025/26 |
| [Transfermarkt](https://transfermarkt.com) | Squad market values per team per season | 2016/17 → 2025/26 |
| [football-data.co.uk](https://football-data.co.uk) | Bet365 bookmaker odds | 2024/25 |

---

## Methodology

### 1. Exploratory Data Analysis
- Distribution of H/D/A outcomes across seasons
- Home advantage analysis — including the Covid-19 effect (2020/21, behind closed doors)
- xG vs actual goals correlation — validating xG as a reliable performance metric
- PPDA analysis — pressing intensity as a proxy for team style

### 2. Feature Engineering

All features are computed **strictly from data available before each match** to prevent data leakage.

| Feature category | Examples |
|-----------------|---------|
| Season standings | `points_diff`, `home_points`, `away_points` |
| Rolling form (last 5 matches) | `rolling_xG_diff`, `home_rolling_winrate`, `rolling_ppda_diff` |
| Home/Away split performance | `home_rolling_xg_home_only`, `away_rolling_xg_away_only` |
| Squad quality | `squad_value_ratio`, `squad_value_diff` |
| Differentials | `rolling_goals_diff`, `rolling_winrate_diff` |

### 3. Modelling

Three models trained on 8 seasons (2016/17 → 2023/24), tested on 2024/25 (211 matches) :

| Model | Accuracy | Log-loss |
|-------|----------|----------|
| Logistic Regression (baseline) | 0.564 | 0.965 |
| **Random Forest** | **0.569** | **0.957** |
| XGBoost | 0.550 | 1.000 |

**Train/test split is temporal** — the model is always trained on past seasons and tested on a future season, mimicking real production conditions.

### 4. Bookmaker comparison

Model probabilities are compared against Bet365 implied probabilities (normalized to remove the ~5.5% overround margin) on the 2024/25 test season.

Key findings :
- **Home/Away** : model broadly aligns with bookmaker pricing — market is efficient
- **Draws** : model assigns compressed, near-uniform probabilities (~25%) regardless of context — the main weakness and a potential source of market inefficiency

---

## Key results

- **Best model** : Random Forest — 56.9% accuracy, log-loss 0.957
- **Naive baseline** (bet on team with more points) : 49.3% accuracy → model adds **+7.6 points**
- **Bookmaker accuracy** : ~53-56% on European leagues (academic literature)
- **Main limitation** : draws are never predicted (0 out of 39 actual draws correctly classified)
- **Top features** : `squad_value_ratio` > `squad_value_diff` > `points_diff` > `rolling_xG_diff`

---

## Limitations and next steps

- **Draw prediction** : the model fails entirely on draws — future work should explore draw-specific features (H2H records, ELO ratings, defensive rigidity)
- **No real-time features** : injuries, suspensions, and lineup information are not included — bookmakers price these immediately
- **ELO rating** : a dynamic ELO system would better capture team quality across promotions/relegations than season-static squad values
- **Multi-league extension** : adding Premier League, La Liga, Bundesliga would multiply training data ~5x and likely unlock XGBoost's full potential

---

## Requirements
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn openpyxl
```

---

## Author

Baptiste — M1 Finance, EDHEC Business School / École Centrale de Lille  
Project built as part of a quantitative finance curriculum, with a focus on the intersection of machine learning and financial market analysis.
