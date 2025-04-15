# Employee Attrition Modeling (ML)

This project demonstrates how to build and evaluate machine learning models to predict employee attrition using real-world HR data. The focus is on applying practical, interpretable machine learning to a business-critical problem, showcasing both model accuracy and insights for HR decision-making.

---

## Project Highlights

- **Real-world HR dataset** from IBM (via open-source GitHub repo)
- Full ML pipeline: load → clean → encode → split → model → interpret
- **Logistic Regression**: Solver tuning + class weight handling
- **Random Forest**: Non-linear modeling + feature importance analysis
- Emphasis on **interpretability**, **business logic**, and **model comparison**

---

## Key Questions Answered

- What features most strongly predict employee attrition?
- How does overtime, travel, or job role affect retention?
- How do different models perform in terms of recall, precision, and overall utility?

---

## Models Used

### 1. Logistic Regression
- Solver: `liblinear`
- Class weight: `balanced`
- Strength: High recall on attrition (catching at-risk employees)
- Interpretation: Feature coefficients ranked by influence

### 2. Random Forest Classifier
- Ensemble of 100 decision trees
- Class weight: `balanced`
- Strength: High overall accuracy, better precision
- Interpretation: Feature importance ranking from trained model

---

## Results Snapshot

| Metric           | Logistic Regression | Random Forest     |
|------------------|---------------------|-------------------|
| Accuracy         | 71%                 | **87%**           |
| Recall (class 1) | **54%**             | 8%                |
| Precision (1)    | 24%                 | **75%**           |
| F1-score (1)     | **0.33**            | 0.14              |

---

## Feature Insights

| Feature                    | Model Weight (Direction) |
|----------------------------|---------------------------|
| `OverTime_Yes`             | ↑ Attrition               |
| `BusinessTravel_Travel_Frequently` | ↑ Attrition     |
| `JobRole_Laboratory Technician` | ↑ Attrition       |
| `WorkLifeBalance`          | ↓ Attrition               |
| `EducationField_Medical`   | ↓ Attrition               |
| `YearsInCurrentRole`       | ↓ Attrition               |

---

## Structure

employee-attrition-modeling-ml/ │ ├── data/ # Source or local datasets ├── notebooks/ # (Optional) Jupyter visuals ├── scripts/ # Main .py file(s) │ └── employee_attrition_model.py ├── outputs/ # Charts, model logs, output files ├── README.md # This file └── requirements.txt # Dependencies

---

## Getting Started

1. Clone the repository:

```bash

git clone https://github.com/yourusername/employee-attrition-modeling-ml.git

Install dependencies:

pip install -r requirements.txt

Run the script:

python scripts/employee_attrition_model.py

Future Additions
Add SHAP explanations or LIME for deeper interpretability

Model deployment via Streamlit or Flask

Integration with Power BI or Dash for HR teams

Author Notes
This project was created as part of a daily AI/ML fluency challenge. The emphasis is on applying core ML skills to realistic business use cases, building a personal GitHub portfolio, and reinforcing understanding through comparison, interpretation, and storytelling.
