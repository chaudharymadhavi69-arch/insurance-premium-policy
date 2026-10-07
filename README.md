# Insurance Premium Prediction

A machine-learning project that estimates insurance premiums from customer attributes such as age, BMI, smoking status, children, sex and region.

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- SQL
- Streamlit

## Workflow
1. Load and inspect the dataset.
2. Separate features and target (`charges`).
3. Preprocess categorical variables using One-Hot Encoding.
4. Split data into training and testing sets.
5. Train a Random Forest Regression model.
6. Evaluate using MAE, RMSE and R².
7. Save the trained pipeline with Joblib.
8. Use Streamlit for interactive prediction.

## Run Locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/model.py
streamlit run app.py
```

Open the Streamlit URL shown in the terminal.

## Important note about "92% accuracy"
This is a regression project, so plain classification-style accuracy is not the correct primary metric. The training script reports R², MAE and RMSE. Use the actual measured R² from your run before putting a percentage claim on your resume.
