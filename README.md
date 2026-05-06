# Early Suicide Prediction using Machine Learning

## Project Overview
This project investigates whether supervised machine learning models can predict **Self Harm Story** using structured student mental health data. The study compares three classifiers:

- Gaussian Naive Bayes
- Support Vector Machine (SVM)
- Random Forest

The goal is to identify which model performs best for this classification task.

## Research Question
To what extent can supervised machine learning models, specifically Naive Bayes, Support Vector Machine (SVM), and Random Forest, accurately classify self-harm behaviour from student mental health data, and which model provides the best predictive performance based on accuracy, recall, and F1-score?

## Aim
To develop, implement, and compare supervised machine learning models for predicting self-harm behaviour from student mental health data in order to identify the most effective classifier.

## Dataset
The project uses the **Early Suicide Prediction** dataset from Kaggle.

### Dataset Details
- Initial dataset shape: **(1099, 12)**
- Duplicate rows removed: **45**
- Final cleaned dataset shape: **(1054, 12)**
- Missing values in **Family Problem**: **379**
- Missing values handled by replacing them with **None**

### Variables
The dataset includes the following variables:
- Age
- Gender
- Stress Level
- Academic Performance
- Health Condition
- Relationship Status
- Family Problem
- Depression Level
- Anxiety Level
- Mental Support
- Self Harm Story
- Suicide Attempt

## Project Workflow
The project followed these main steps:

1. Data loading and initial inspection  
2. Duplicate removal  
3. Missing-value handling  
4. Text cleaning  
5. Label encoding  
6. Train-test split  
7. Class imbalance handling using **SMOTE**  
8. Feature scaling using **StandardScaler**  
9. Model training and evaluation  
10. Hyperparameter tuning using **RandomizedSearchCV**

## Data Splitting and Balancing
- Training feature shape: **(843, 11)**
- Testing feature shape: **(211, 11)**
- Training target shape before SMOTE: **(843,)**
- Training target shape after SMOTE: **(1358,)**

## Models Used
### 1. Gaussian Naive Bayes
Baseline parameter:
- `var_smoothing = 1e-9`

Best tuned parameter:
- `model__var_smoothing = 3.856620421163472e-07`

### 2. Support Vector Machine (SVM)
Baseline parameters:
- `kernel='rbf'`
- `C=1.0`
- `gamma='scale'`
- `probability=True`
- `random_state=42`

Best tuned parameters:
- `model__kernel='rbf'`
- `model__gamma=0.07498942093324558`
- `model__C=0.03162277660168379`

### 3. Random Forest
Baseline parameters:
- `n_estimators=200`
- `max_depth=10`
- `min_samples_split=5`
- `min_samples_leaf=2`
- `random_state=42`
- `n_jobs=-1`

Best tuned parameters:
- `model__n_estimators=100`
- `model__min_samples_split=2`
- `model__min_samples_leaf=1`
- `model__max_features='log2'`
- `model__max_depth=5`
- `model__class_weight='balanced'`

## Evaluation Metrics
The models were evaluated using:
- Accuracy
- Recall
- F1-score
- Classification report
- Confusion matrix
- ROC curve and AUC

## Results

| Model | Accuracy | Recall | F1-score |
|------|---------:|-------:|---------:|
| Naive Bayes | 0.905213 | 0.951220 | 0.795918 |
| SVM | 0.895735 | 0.902439 | 0.770833 |
| Random Forest | 0.919431 | 0.853659 | 0.804598 |

### Key Findings
- **Random Forest** achieved the best overall performance with the highest **accuracy** and **F1-score**.
- **Naive Bayes** achieved the highest **recall**, making it strongest for identifying positive cases.
- **SVM** also performed well, but its overall scores were slightly lower than the other two models.

## Hyperparameter Tuning Results

| Model | Tuning Method | Best CV F1-score |
|------|---------------|-----------------:|
| Gaussian Naive Bayes | RandomizedSearchCV | 0.7188781884067152 |
| SVM | RandomizedSearchCV | 0.776347119844339 |
| Random Forest | RandomizedSearchCV | 0.784602670990741 |

## Visual Outputs
The project includes:
- Class distribution plot
- Gender vs Self Harm Story plot
- Stress level distribution plot
- Correlation heatmap
- Confusion matrices
- ROC curve comparison
- Model performance comparison chart

## Repository Structure
```text
project/
│
├── data/
│   └── Final_SP_dataSet.csv
│
├── notebooks/
│   └── final_project_notebook.ipynb
│
├── figures/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── model_comparison.png
│
├── report/
│   └── final_report.pdf
│
└── README.md

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

How to Run
Download or clone the repository
Place the dataset file in the correct folder
Open the notebook
Run all cells in sequence
