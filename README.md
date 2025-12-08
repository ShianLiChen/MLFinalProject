# CSCI 4050 - ML Final Project (ID: Group 16)
## By: Li Chen (100813628) &

# Sections:
1. Files whose names end in `*_dgg` represent the **Diabetes General Guess** functionality that uses a Multi-Layer Perceptron (MLP) and 2 multi-participant diabetic information datasets to predict whether a user has diabetes and also provides a confidence level in its prediction.
2. Files whose names end in `*_idg` represent the **Insulin Dosage Guess** functionality that uses a multi-input MLP and a single participant sequential diabetic information tracking dataset to predict and suggest the user's next insulin dosage. 

# Learning Problem: 
The purpose of this project is to create a Machine Learning model that can predict whether you have diabetes or not and also give you the percentage of confidence that the prediction is accurate. It uses various information including: demographic information (gender, age, number of pregnancies), physical measurements (BMI, blood pressure, skin thickness), lab measurements (blood glucose, HbA1c, insulin) and health history (hypertension, heart disease, smoking status) to help make the predictions. The model was trained using 2 datasets taken from Kaggle and uses a sequence of linear and ReLu layers to help conduct a binary classification of whether a user has diabetes. The model also uses the PyTorch Lightning module and the Adam optimizer to help improve the model's performance.

## Motivation:
Early detection of diabetes is crucial as it allows for the initiation of long-term accurate health management and prevents further deteriation in user lifetsyle due to diabetic symptoms. Machine learning can help facilitate this detection by offering fast and cost-effective preliminary risk assessments prior to clinical consultation. Our goal is to build a deployable prediction model that is also integrated in a web application for easier user utilization. 

# Data:
2 datasets were taken from Kaggle: 
* The first dataset is from: [Diabetes Data 1](https://www.kaggle.com/datasets/mathchi/diabetes-data-set?resource=download)
* The second dataset is from: [Diabetes Data 2](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

## Data Processing:
The two datasets were merged after comparing and matching shared feature column names. Missing values were handled by substituting median values into the missing columns allowing for maintenance of a large sample size. In order to help with model functionality, continuous features were also standardized and categorical fields were encoded to numerical representations.

# Deployment:
The trained PyTorch models have been deployed through a web application built using Flask, python and Bootstrap. Users are presented with a web form, upon application load, where they can enter demographic, physical, lab, and lifestyle information. Once the form is submitted, the model returns both a diabetes prediction and confidence level. The application handles the preprocessing of the user's input including normalization of continuous values and numerical encoding of categorical variables. The data is then sent along with the feature scaling values (`mean.npy` and `scale.npy`) to the trained model (`model.pt`), where the prediction is generated. This Flask application demonstrates how the model can be integrated into a web form application with a user-friendly user interface and removes the need for the user to train the model.

# Running the Application:
**Architecture:** Flask, Python, Pytorch, Lightning, Bootstrap, HTML, JS
1. Clone the GitHub repo to your local drive
2. After cloning the repo to your local drive, open a terminal and navigate to the project folder (where `requirements.txt` is located).
3. Create and activate a virtual environment:
```
python -m venv venv
venv\Scripts\activate
```
4. Install the dependencies using: `pip install -r requirements.txt`
5. Prior to running the Flask application ensure that you have the `saved_model\` folder, containing the `mean.npy`, `model.pt` and `scale.npy` files.
    * If these files are not copied to your local drive you can either redownload and try again or do the following:
    * In the same root folder run: `python -m src.train` (Windows Command Prompt) or the equivalent on your operating system
    * This command should generate the required `saved_model\` files
6. Once the required files are in the `saved_model\` location, in the same root folder in the terminal run: `python app/app.py`
7. You can now access the application form through the `http://127.0.0.1:5000/` link on your browser of choice. 
8. Fill out the fields on the form and once finished, click the predict button near the bottom of the form. 
9. The model will return its diabetes prediction and the probability that it is confident in its decision.
