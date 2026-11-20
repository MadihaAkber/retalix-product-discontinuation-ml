# Product Discontinuation Prediction Using Machine Learning

## Project Overview
An end-to-end machine learning project developed as part of the MSc Applied Artificial Intelligence programme at the University of Westminster.

The project predicts whether a retail product is likely to be discontinued using historical catalogue data.

## Technologies
Python, Pandas, NumPy, Scikit-learn, Logistic Regression, Random Forest, GridSearchCV, RandomizedSearchCV, Matplotlib, Seaborn, Pytest and Streamlit.

## Machine Learning Workflow
1. Data ingestion and cleaning
2. Exploratory data analysis
3. Feature engineering
4. Chronological training and testing
5. Model training and hyperparameter tuning
6. Performance evaluation
7. Threshold optimisation
8. Simulated model monitoring
9. Streamlit web application

## Dataset
- Training editions: 88–92
- Test edition: 93
- Prediction snapshot: Week −24

Raw datasets are excluded from GitHub.

## Final Model
**Tuned Logistic Regression**

Decision threshold: 0.10

### Test Results
- Accuracy: 86.48%
- Precision: 77.80%
- Recall: 80.12%
- F1-score: 78.94%

## Installation
Install the dependencies using:

`pip install -r requirements.txt`

## Testing
Run automated tests using:

`python -m pytest tests/ -v`

## Streamlit Application
Launch the application using:

`python -m streamlit run app.py`

The application requires the trained model file, which is not included in the GitHub repository.

## Limitations
- Predictions require an available Week −24 snapshot.
- Missing snapshots may introduce selection bias.
- Performance may vary across catalogue editions.
- Monitoring is simulated rather than live.

## Author
Madiha Fatima Agha

MSc Applied Artificial Intelligence

University of Westminster