# Emotion Classification in Text
## Objective
The goal of this project is to develop machine learning models for classifying emotions in text samples. The project involves several steps, including loading and preprocessing the dataset, feature extraction, model development, and evaluation.

## Dataset
The dataset used for this project can be found here. It contains text samples labeled with corresponding emotions.
https://drive.google.com/file/d/1HWczIICsMpaL8EJyu48ZvRFcXx3_pcnb/view?usp=drive_link

## Key Components
### 1. Loading and Preprocessing 
The dataset is loaded and preprocessed to ensure that it is in a clean and suitable format for machine learning models. This includes the following steps:

#### Text Cleaning: Removal of unwanted characters, symbols, and extra spaces.
Tokenization: Breaking the text into individual words or tokens.
Stopword Removal: Removing common but irrelevant words (such as "the", "is", "and", etc.) to improve the model’s performance.
These preprocessing steps help the models focus on the meaningful parts of the text, improving classification accuracy and reducing overfitting.

### 2. Feature Extraction 
To convert the textual data into numerical form, we used the TfidfVectorizer. This method transforms the text data into a matrix of token counts weighted by term frequency-inverse document frequency (TF-IDF). The TF-IDF vectorization helps in capturing the importance of words based on their frequency and relevance across the entire corpus.

TF-IDF works well because it takes into account both the frequency of the word in a document and how unique the word is across all documents, which is useful for emotion classification, where context-specific words matter.
### 3. Model Development 
The following machine learning models were trained on the dataset:

#### Naive Bayes: A probabilistic classifier based on Bayes' theorem, which works well for text classification tasks, particularly with large datasets.
#### Support Vector Machine (SVM): A discriminative classifier that aims to find the hyperplane that best separates different classes in the feature space.
Both models are commonly used for text classification due to their efficiency and robustness in handling high-dimensional data.

### 4. Model Comparison 
The models were evaluated using the following metrics:

#### Accuracy: The ratio of correctly predicted instances to the total instances.
#### F1-Score: The harmonic mean of precision and recall, providing a balance between the two.
These metrics were chosen because they are standard evaluation measures for classification tasks and are particularly useful when dealing with imbalanced datasets, as they consider both false positives and false negatives.

## Conclusion
This project demonstrates the use of machine learning techniques for emotion classification in text. By preprocessing the data, extracting relevant features, and training multiple models, we were able to compare their performance and select the best approach for the task.

#### Files Included
emotion_classification.ipynb: Jupyter notebook containing the full implementation.
requirements.txt: A list of Python packages required to run the code.
dataset: Link to the dataset used for the project.

