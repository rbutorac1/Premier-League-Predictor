# Premier-League-Predictor

**⚽ Premier League Prediction & Monte Carlo Simulator**

This project implements a full football analytics pipeline for the English Premier League, combining:

* **Multi-target XGBoost match prediction**
* **Poisson goal modelling** 
* **Dynamic ELO rating system**
* **Monte Carlo season simulation**

---

**🔍 Match-Level Prediction**

The model predicts detailed match statistics including:

* Goals
* Shots & Shots on Target
* Corners
* Yellow & Red Cards

Expected goals (λ_home, λ_away) are fed into a Poisson model to compute match outcome probabilities (Home Win / Draw / Away Win).

**Poisson distribution formula:**

\[
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}
\]

where:

- \( \lambda \) = expected goals  
- \( k \) = number of goals scored  

---

**📈 Dynamic ELO System**

Team strength evolves during simulation using an ELO rating update mechanism with home advantage and configurable K-factor.

**ELO update formula:**

\[
R_{new} = R_{old} + K (S - E)
\]

where:

- \( R_{old} \) = previous rating  
- \( K \) = update factor  
- \( S \) = actual result (1 = win, 0.5 = draw, 0 = loss)  
- \( E \) = expected result  

Expected result calculation:

\[
E = \frac{1}{1 + 10^{(R_{opponent} - R_{team}) / 400}}
\]

---

**🎲 Season Simulation (Monte Carlo)**

Using a lightweight ELO-based model, the engine simulates the full 380-match season thousands of times to estimate:

🏆 Title probability  
🔝 Top 4 probability  
⬇ Relegation probability  
📊 Expected final position  
📈 Expected points  

Results are derived from a full 20×20 finishing-position probability matrix.

---

**🧠 Purpose**

This project demonstrates how to combine:

* Machine Learning
* Probabilistic modelling
* Rating systems
* Monte Carlo simulation

into a complete football forecasting framework.
