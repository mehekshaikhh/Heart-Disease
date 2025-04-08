
This project uses machine learning to predict whether a person is at risk of heart disease based on various health metrics. The model was built using the UCI Heart Disease dataset and trained on various machine learning algorithms to provide predictions that could assist in early diagnosis and preventive healthcare.

# Technologies Used
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook


# Dataset Information

The dataset used is the UCI Heart Disease dataset, which contains data on various health features such as age, blood pressure, cholesterol, and maximum heart rate. The target variable is `HeartDisease` (1 if the person has heart disease, 0 otherwise).

# Features:
- Age
- RestingBP
- Cholesterol
- FastingBS
- MaxHR
- Oldpeak
- Sex_M (One-hot encoding of Sex)
- ChestPainType (One-hot encoding)
- ExerciseAngina (One-hot encoding)
- RestingECG (One-hot encoding)
- ST_Slope (One-hot encoding)

# Steps to Run

# 1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
# 2. Install dependencies: Create a requirements.txt file and add the following libraries:
pandas,numpy,scikit-learn,matplotlib,seaborn,jupyter
 Then, run:
pip install -r requirements.txt
# 3:Run the Jupyter notebook: Start Jupyter Notebook to run the code:
jupyter notebook heart_disease_prediction.ipynb
# 4: Load the trained model (if necessary for prediction):
import pickle
model = pickle.load(open('heart_model.pkl', 'rb'))
# 5: To make predictions with your model:
y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
