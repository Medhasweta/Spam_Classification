# Spam_Classification
## Spam Detection in Python using NLP and Machine Learning
This project demonstrates the application of Natural Language Processing (NLP) and various machine learning algorithms in Python to classify messages as spam or ham (not spam). The project uses popular Python libraries such as NumPy, Pandas, scikit-learn, and NLTK.

## Dependencies
NumPy
Pandas
scikit-learn
NLTK
Data
The dataset used is a collection of SMS messages, labeled as spam or ham. The dataset is read from a CSV file and preprocessed for analysis.

## Preprocessing Steps
Label Encoding: Convert class labels (spam/ham) into binary values (0/1).
Text Processing:
Replace email addresses, URLs, phone numbers, money symbols, and other numbers with specific placeholders.
Remove punctuation and excess whitespace.
Convert all text to lowercase.
Remove stop words.
Apply stemming to reduce words to their root form.
Tokenization: Tokenize the text to create a bag-of-words model.
Feature Selection: Select the most common words as features for the model.

## Machine Learning Models
Multiple machine learning models are trained and evaluated:

Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Decision Tree
Random Forest
Logistic Regression
Voting Classifier: An ensemble method combining the above models.
Evaluation
The models are trained on a training set and evaluated on a test set. The performance is measured in terms of accuracy, confusion matrix, and classification report.

## Usage
Import the required libraries and the dataset.
Perform data preprocessing.
Create bag-of-words and select features.
Split the data into training and testing sets.
Train and evaluate the machine learning models.
Use the best-performing model to classify new messages.
Notes
Ensure the dataset path in the code matches the location of your CSV file.
You may need to download specific resources from NLTK, like stopwords and punkt.
