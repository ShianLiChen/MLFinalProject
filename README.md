# CSCI 4050 - ML Final Project (ID: Group 16)
## By: Li Chen (100813628) &

# Sections:
1. Files whose names end in `*_dgg` represent the **Diabetes General Guess** functionality that uses a Multi-Layer Perceptron (MLP) and 2 multi-participant diabetic information datasets to predict whether a user has diabetes and also provides a confidence level in its prediction.
2. Files whose names end in `*_idg` represent the **Insulin Dosage Guess** functionality that uses a multi-input MLP and a single participant sequential diabetic information tracking dataset to predict and suggest the user's next insulin dosage. 

# Section 1:
## Learning Problem: 
The purpose of this section of the project is to create a Machine Learning model that can predict whether you have diabetes or not and also give you the percentage of confidence that the prediction is accurate. It uses various information including: demographic information (gender, age, number of pregnancies), physical measurements (BMI, blood pressure, skin thickness), lab measurements (blood glucose, HbA1c, insulin) and health history (hypertension, heart disease, smoking status) to help make the predictions. The model was trained using 2 datasets taken from Kaggle and uses a sequence of linear and ReLu layers to help conduct a binary classification of whether a user has diabetes. The model also uses the PyTorch Lightning module and the Adam optimizer to help improve the model's performance.

### Motivation:
Early detection of diabetes is crucial as it allows for the initiation of long-term accurate health management and prevents further deteriation in user lifetsyle due to diabetic symptoms. Machine learning can help facilitate this detection by offering fast and cost-effective preliminary risk assessments prior to clinical consultation. Our goal is to build a deployable prediction model that is also integrated in a web application for easier user utilization. 

## Data:
2 datasets were taken from Kaggle: 
* The first dataset is from: [Diabetes Data 1](https://www.kaggle.com/datasets/mathchi/diabetes-data-set?resource=download)
* The second dataset is from: [Diabetes Data 2](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

### Data Processing:
The two datasets were merged after comparing and matching shared feature column names. Missing values were handled by substituting median values into the missing columns allowing for maintenance of a large sample size. In order to help with model functionality, continuous features were also standardized and categorical fields were encoded to numerical representations.

# Section 2:
## Learning Problem: 
The purpose of this section of the project is to create a Machine Learning model that can predict the amount of insulin a person should administer at a given time based on their recent glucose levels, carbohydrate intake, and other relevant health factors. The model uses various types of information, including time of day, blood glucose readings, carbohydrate consumption, previous insulin doses, and meal events to provide a personalized insulin recommendation. The model is trained using a dataset derived from real user insulin and diabetes logs, which has been preprocessed and structured as time-series sequences. It uses MLP architecture with linear and ReLU layers to learn complex relationships between the input features and the target insulin amount. The model also uses the PyTorch Lightning module and the Adam optimizer to help improve the model's performance. 

### Motivation:
As insulin intake is an essential part of maintaining regular health for diabetics, ensuring the correct insulin dosage is taken is essential to prevent immediate adverse health declines. Although, currently, there are insulin calculation methods, it is still very much a manual process. Integrating Machine Learning will help not only automate this calculation but will also allow for more accurate dosing as historical data can leveraged to fine tune the ML model, thus allowing for a more personalized insulin dosage. Our goal in this section is to use user data to build a model that accurately predicts the amount of insulin the user should be taking based on user inputs like amount of carbohydrates eaten, current blood glucose, target blood glucose and time of day. 

## Data:
The dataset used for this part of the application is based off of diabetic tracking information logged by a single user. 

### Data Processing:
The insulin dataset was first cleaned and formatted by encoding timestamps into cyclical features (sin/cos) as it represents the time of day and cyclical nature of the data. Missing values for insulin amount taken were filled with the recommended dose rounded to the nearest 0.5 to preserve dataset size. Categorical features like meal events were encoded numerically to match breakfast, lunch, dinner or workout and monthly average glucose was calculated to help provide the model with meaningful inputs.

# Tertiary/Inherent Learning Problems
During the process of creating both the models for section 1 and 2 a lot of new information was realized. The first being the method of implementing median or target inputs for missing values can lead to biasing of a machine learning model and false inflation of the accuracy level. The second being how training data can significantly impact predictions and leads to biasing of preditcions that tend to favour the output that is more frequent in the data. This is seen through the diabetes diagnosis prediction model that favours the non-diabetic prediction even with input values that would strongly suggest the user has diabetes, caused by the data having significantly more non-diabetic rows than diabetic rows. Lastly, the difference in model creation and evaluation varies greatly with the type of data (e.g. timeseries vs single user input). It is easier to determine accuracy and requires much less epoch iteration for single user input compared to timeseries data. 

# Deployment:
**Architecture:** Flask, Python, Pytorch, Lightning, Bootstrap, HTML, JS
The trained PyTorch models have been deployed through a web application built using Flask, python and Bootstrap. Upon application load, users are presented with a web form for predicting diabetic diagnosis, where they can enter demographic, physical, lab, and lifestyle information. Once the form is submitted, the model returns both a diabetes prediction and confidence level. The application handles the preprocessing of the user's input including normalization of continuous values and numerical encoding of categorical variables. The data is then sent along with the feature scaling values (`mean_dgg.npy` and `scale_dgg.npy`) to the trained model (`model_dgg.pt`), where the diabetes diagnosis prediction is generated. 

The user can also access the insulin prediction page through the top navigation bar. On this page users will be required to enter 4 inputs: amount of carbohydrates eaten, current blood glucose level, target blood glucose level and the current time of day. These fields are by default populated with default average values. Once the form is submitted, the data is preprocessed to represent the time as part of a cyclical feature and is sent to the stored trained model (`model_idg.pt`), where the insulin dosage prediction is generated. 

This Flask application demonstrates how both models can be integrated into a web application with a user-friendly user interface and removes the need for the user to train the model.

# Running the Application:
1. Clone the GitHub repo to your local drive
2. After cloning the repo to your local drive, open a terminal and navigate to the project folder (where `requirements.txt` is located).
3. Create and activate a virtual environment:
```
python -m venv venv
venv\Scripts\activate
```
4. Install the dependencies using: `pip install -r requirements.txt`
5. Prior to running the Flask application ensure that you have the `saved_model\` folder, containing the `mean_dgg.npy`, `model_dgg.pt`, `model_idg.pt` and `scale_dgg.npy` files.
    * If these files are not copied to your local drive you can either redownload and try again or do the following:
    * In the same root folder run: `python -m src.train_dgg` (Windows Command Prompt) or the equivalent on your operating system
    * This command should generate the required `saved_model\` `*_dgg` files
    * For the `*_idg` files you first need to run `python -m src.data.clean_data_idg` and then `python -m src.train_idg`
6. Once the required files are in the `saved_model\` location, in the same root folder in the terminal run: `python app/app.py`
7. You can now access both application forms through the `http://127.0.0.1:5000/` link on your browser of choice. 
8. Fill out the fields on either form and once finished, click the predict button near the bottom of the form. 
9. The model will return its prediction and the probability that it is confident in its decision (for the diabetes diagnosis prediction).
