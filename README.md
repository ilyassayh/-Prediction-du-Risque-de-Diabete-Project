# Diabetes Risk Prediction Project / Projet de Prédiction du Risque de Diabète

## Overview / Présentation

This project aims to predict diabetes risk using machine learning techniques on a medical dataset. The workflow covers data cleaning, exploratory analysis, clustering, classification, and model evaluation, with a focus on explainability and reproducibility.

Ce projet vise à prédire le risque de diabète à l'aide de techniques de machine learning sur un jeu de données médical. Le workflow comprend le nettoyage des données, l'analyse exploratoire, le clustering, la classification et l'évaluation des modèles, avec un accent sur l'explicabilité et la reproductibilité.

---

## Files / Fichiers
- `dataset-d.csv`: Main dataset used for analysis / Jeu de données principal.
- `modele_de_Prédiction_du_Risque_de_Diabète.ipynb`: Main notebook with the full workflow / Notebook principal avec tout le workflow.
- `notebook_with_full_explanations.ipynb`: Notebook enriched with detailed explanations after each code cell / Notebook enrichi avec des explications détaillées.
- `insert_markdown_explanations.py`: Script to insert explanations into the notebook / Script pour insérer des explications dans le notebook.
- `best_xgboost_model.pkl`: Saved XGBoost model / Modèle XGBoost sauvegardé.

---

## Dataset
- 768 rows, 8 features (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
- Medical data for diabetes risk prediction

---

## Workflow
1. **Data Loading & Exploration**: Load and inspect the dataset, check for missing values and outliers.
2. **Data Cleaning**: Replace zeros with NaN, impute missing values, remove outliers using IQR.
3. **Visualization**: Histograms, correlation heatmap, pairplots for feature relationships.
4. **Clustering**: K-Means clustering and PCA for dimensionality reduction and visualization.
5. **Classification**: Train/test split, oversampling, model training (Random Forest, SVM, Gradient Boosting, Logistic Regression, XGBoost), hyperparameter tuning.
6. **Evaluation**: Confusion matrix, classification report, cross-validation, model saving.

---

## Results
- Significant outlier removal (from 768 to 332 rows)
- XGBoost selected as the best model (high F1 score, good generalization)
- Clusters identified for risk stratification

---

## Visualizations
- Histograms for feature distributions
- Correlation heatmap
- Pairplots for feature relationships
- Cluster visualizations (original space and PCA)
- Confusion matrix for model performance

---

## How to Run / Comment exécuter
1. Install dependencies: `pip install -r requirements.txt` (requirements: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, nbformat)
2. Run the main notebook: `modele_de_Prédiction_du_Risque_de_Diabète.ipynb` or `notebook_with_full_explanations.ipynb`
3. To generate the enriched notebook: run `insert_markdown_explanations.py`

---

## Author / Auteur
Ilyas sayh

---

## License
MIT 
