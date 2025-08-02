# AI Corners Prediction Model

This repository contains a fully functional binary classification model using XGBoost to predict whether a football match will have **over or under 11.5 total corners**.

---

## 🚀 What It Does

- Trains on real match data from 2016–2025 using the API-Football service
- Predicts whether a match will have **12+ corners (OVER)** or **11 or fewer (UNDER)**
- Uses advanced features such as:
  - Shot and possession efficiency
  - Corner-to-attack ratios
  - SHAP value explanations
  - Permutation-based feature importance
- Automatically evaluates upcoming matches
- Builds **parlay suggestions** based on high-confidence predictions

---

## 🧠 Betting Strategy Overview

- Each parlay is composed of **2 matches**
- It's recommended to bet in **blocks of 50 parlays (≈100 matches)** to smooth out variance
- The model has been **validated on 90+ parlays (~180 matches)** with real match data
- Expected win rate: **64%**
  - Historical range: **59% to 69%**, depending on match conditions
- **Unprofitable leagues have been removed** from prediction (as shown in the images)
- All predictions are based on historical performance; results are not guaranteed

---

## 🎯 How the Model Chooses Parlays

The model only creates a parlay if the **combined calibrated probability is ≥ 64%**.

How calibration works:
1. Let's say match A has an 85% raw probability. The calibration map says this maps to 89%.
2. Match B has an 80% raw probability. Calibrated: 80%.
3. Combined: 0.89 × 0.80 = **71.2% calibrated success rate**, so it's included in the list.

Thresholds are derived from real match validations using historical accuracy at different probability levels.

---

## 📂 Files

- `train_model.py`: Model training, evaluation, and prediction on upcoming matches
- `requirements.txt`: Python dependencies
- `README.md`: You're reading it

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

Make sure to insert your API key in the script:
```python
API_KEY = "YOUR_API_KEY"
```

---

## 🔐 Notes

- This model uses public match data from the RapidAPI Football API
- Make sure your plan supports historical and fixture stats endpoints
- All results are empirical, based on real matches and league performance
- The script can be used for research, model improvement, or practical betting strategies

---

## 🧠 Disclaimer

**This is not financial advice.**  
Sports outcomes are inherently uncertain. While this model is built and validated on real data, no outcome is guaranteed. Always bet responsibly.

---

## 📬 Contact

For commercial licensing, questions or collaborations:  
📧 carlosazarate13@gmail.com


---

## 🔧 Tips & Customization

### Changing League or Country
By default, the model uses the English Premier League (league ID 135).  
To change this, modify the line in `train_model.py`:

```python
params = {"league": 135, "season": season}
```

You can find league IDs from the API-Football documentation or use the "leagues" endpoint.

---

### Handling API Errors
If the script doesn't return results, check the following:
- You may have reached your API call limit.
- Your API key might be incorrect or expired.
- There may be no upcoming matches scheduled (`status.short == 'NS'`).
- A league might not have sufficient data for predictions.

---

### Output Format
All predictions and results are printed to the console:
- Probabilities for UNDER and OVER
- Match recommendations
- Calibrated parlay combinations

**Sample output:**

```
📅 2025-05-20 - Chelsea vs Everton
  🔮 UNDER 11.5 Probability: 83.1%
  🔮 OVER 11.5 Probability: 16.9%
  ✅ RECOMMENDED: Bet UNDER 11.5 with high confidence

✅ UNIQUE Parlays with calibrated combined probability ≥ 64%:
  • Chelsea vs Everton + PSG vs Lyon → 71.2%
```

---

### Saving the Model (Optional)
You can save the trained model for reuse with:

```python
import joblib
joblib.dump(model, "corner_model.pkl")
```

And later load it with:

```python
model = joblib.load("corner_model.pkl")
```

This avoids re-training every time you want to predict.

---

### Optional: Export to CSV
If you'd like to export the predictions instead of just printing them:

```python
df_predictions.to_csv("predictions.csv", index=False)
```

---

## AWS Artifacts & Planned Deployment

- **Region:** us-east-1 (N. Virginia)
- **S3 (private, versioned):** Release artifacts at `s3://corners-ml-artifacts-carlos-20250802-usae1/releases/`
- **Infrastructure as Code:** [`infra/template.yaml`](infra/template.yaml) (CloudFormation) describing a planned deployment:
  - S3 (versioned, private) for data/artifacts
  - IAM Role + Instance Profile for a future EC2-hosted API with S3 read-only access
  - Security Group placeholder for API (HTTP/HTTPS)

> **Note:** This project is not deployed yet. AWS resources are used only for hosting release artifacts (Amazon S3, private and versioned).  
> A CloudFormation template (`infra/template.yaml`) documents a planned deployment path (EC2/API Gateway/Lambda + S3) for future rollout.

## 🔮 Future Work
- Deploy model as a REST API using AWS API Gateway + Lambda or EC2.
- Automate predictions and uploads with a CI/CD pipeline (GitHub Actions → S3).
- Add automated dashboards with match statistics and parlay win-rate tracking.

```mermaid
flowchart LR
  A[Client / Consumer] -->|HTTP| B[(Future API)]
  B -->|Reads data| C[(Amazon S3<br/>releases/, iac/)]
  B -->|Logs artifacts| C
  subgraph Planned Deployment
    B-.->D[EC2 or API Gateway + Lambda]
  end
  C:::s3
classDef s3 fill:#eef,stroke:#88a,stroke-width:1px;



