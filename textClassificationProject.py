import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# data frame for all the news
df = pd.DataFrame(columns=['File_name', 'Title', 'Content', 'Category', 'Class'])

# forming data frame when reading the news from the files
pathB = 'C:/Users/Spale/Desktop/treca godina/drugi semestar/ORI/projekat/datasets/bbc/business'
pathE = 'C:/Users/Spale/Desktop/treca godina/drugi semestar/ORI/projekat/datasets/bbc/entertainment'
pathP = 'C:/Users/Spale/Desktop/treca godina/drugi semestar/ORI/projekat/datasets/bbc/politics'
pathS = 'C:/Users/Spale/Desktop/treca godina/drugi semestar/ORI/projekat/datasets/bbc/sport'
pathT = 'C:/Users/Spale/Desktop/treca godina/drugi semestar/ORI/projekat/datasets/bbc/tech'

def reading_files(df, path, category, class_num, index):
    for file in os.listdir(path):
        with open(path + '/' + file, 'rt') as f:
            title = f.readline()
            content = f.read()
            parts = file.split('.')
            file_name = parts[0]
            index += 1
            df.loc[index] = [file_name, title, content, category, class_num]
    return len(df.index)


i = 0
i = reading_files(df, pathB, 'business', 0, i)
i = reading_files(df, pathE, 'entertainment', 1, i)
i = reading_files(df, pathP, 'politics', 2, i)
i = reading_files(df, pathS, 'sport', 3, i)
i = reading_files(df, pathT, 'tech', 4, i)

print(df)
print(len(df))

# preprocessing
# special character cleaning
df['Title_Parsed_1'] = df['Title'].str.replace("\r", " ")
df['Title_Parsed_1'] = df['Title_Parsed_1'].str.replace("\n", " ")

df['Content_Parsed_1'] = df['Content'].str.replace("\r", " ")
df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("\n", " ")

# lowercasing the text from Title and Content
df['Title_Parsed_2'] = df['Title_Parsed_1'].str.lower()
df['Content_Parsed_2'] = df['Content_Parsed_1'].str.lower()

# removing punstiation signs from Title and Content
punctuation_signs = list('?:!.,;"')
df['Content_Parsed_3'] = df['Content_Parsed_2']
df['Title_Parsed_3'] = df['Title_Parsed_2']
for punct_sign in punctuation_signs:
    df['Content_Parsed_3'] = df['Content_Parsed_3'].str.replace(punct_sign, '')
    df['Title_Parsed_3'] = df['Title_Parsed_3'].str.replace(punct_sign, '')


# lemmatization
# downloading punkt and wordnet from NLTK
#nltk.download('punkt')
#nltk.download('wordnet')

# saving the lemmatizer into an object
wordnet_lemmatizer = WordNetLemmatizer()
num_rows = len(df)
# CONTENT LEMMATIZATION
lemmatized_text_list = []
for row in range(1, num_rows+1):

    # create an empty list containing lemmatized words
    lemmatized_list = []

    # save the text and its words into an object
    text = df.loc[row]['Content_Parsed_3']
    text_words = text.split(" ")

    # iterate through every word to lemmatize
    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))

    # join the list
    lemmatized_text = " ".join(lemmatized_list)

    # append to the list containing the texts
    lemmatized_text_list.append(lemmatized_text)

df['Content_Parsed_4'] = lemmatized_text_list

# TITLE LEMMATIZATION
lemmatized_text_list = []

for row in range(1, num_rows+1):

    # create an empty list containing lemmatized words
    lemmatized_list = []

    # save the text and its words into an object
    text = df.loc[row]['Title_Parsed_3']
    text_words = text.split(" ")

    # iterate through every word to lemmatize
    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))

    # join the list
    lemmatized_text = " ".join(lemmatized_list)

    # append to the list containing the texts
    lemmatized_text_list.append(lemmatized_text)

df['Title_Parsed_4'] = lemmatized_text_list

# reamovig stop words
# downloading the stop words list
#nltk.download('stopwords')
# loading the stop words in english
stop_words = list(stopwords.words('english'))
df['Content_Parsed_5'] = df['Content_Parsed_4']
df['Title_Parsed_5'] = df['Title_Parsed_4']

for stop_word in stop_words:
    regex_stopword = r"\b" + stop_word + r"\b" # mora u ovom formatu
    df['Content_Parsed_5'] = df['Content_Parsed_5'].str.replace(regex_stopword, '')
    df['Title_Parsed_5'] = df['Title_Parsed_5'].str.replace(regex_stopword, '')

print(df.loc[1]['Content_Parsed_5'])

list_columns = ['File_name', 'Title', 'Title_Parsed_5', 'Content', 'Content_Parsed_5', 'Category', 'Class']
df = df[list_columns]
df = df.rename(columns={'Content_Parsed_5': 'Content_Parsed', 'Title_Parsed_5': 'Title_Parsed'})

df['Title_and_Content_Parsed'] = df['Title_Parsed'] + df['Content_Parsed']

print("---------------------------------------------------------------------------------------------------------------")
print(" #### Klasifikacija na osnovu samo naslova vesti #### ")
# forming train and test data
y = df['Class']
X = df['Title_Parsed']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=8)

# Text representation: TF-IDF Vectors
# Parameter election
ngram_range = (1,2) # to consider both unigrams and bigrams
min_df = 10 # when building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold
max_df = 0.8 # when building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold 1
max_features =  2000 # build a vocabulary that only consider the top max_features ordered by term frequency across the corpus

tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)

# fit and transform train features
features_train = tfidf.fit_transform(X_train).toarray()
labels_train = y_train
labels_train = y_train.astype(int)

# only transform test features
features_test = tfidf.transform(X_test).toarray()
labels_test = y_test
labels_test = y_test.astype(int)

# ** Multinomial Naïve Bayes **
print(" * * Multinomial Naïve Bayes * * ")
mnbc = MultinomialNB()
mnbc.fit(features_train, labels_train)
mnbc_pred = mnbc.predict(features_test)

# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(labels_train, mnbc.predict(features_train)))

# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(labels_test, mnbc_pred))

# Classification report
print("Classification report")
print(classification_report(labels_test,mnbc_pred))

# ** Gaussian Naïve Bayes **
print(" * * Gaussian Naïve Bayes * * ")
gnb = GaussianNB()
gnb.fit(features_train, labels_train)
gnb_pred = gnb.predict(features_test)

# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(labels_train, gnb.predict(features_train)))

# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(labels_test, gnb_pred))

# Classification report
print("Classification report")
print(classification_report(labels_test,gnb_pred))

# ** Support Vector Machine **
print(" * * Support Vector Machine * * ")
svm_class = svm.SVC(kernel="linear")
svm_class.fit(features_train, labels_train)
svm_class_pred = svm_class.predict(features_test)

# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(labels_train, svm_class.predict(features_train)))

# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(labels_test, svm_class_pred))

# Classification report
print("Classification report")
print(classification_report(labels_test, svm_class_pred))

# ** Random Forest **
print(" * * Random Forest * * ")
rf = RandomForestClassifier(random_state=0, n_estimators=100, bootstrap=True, max_features="sqrt")
rf.fit(features_train, labels_train)
rf_pred = rf.predict(features_test)

# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(labels_train, rf.predict(features_train)))

# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(labels_test, rf_pred))

# Classification report
print("Classification report")
print(classification_report(labels_test, rf_pred))

# ** K Nearest Neighbors **
print(" * * K Nearest Neighbors * * ")
knn_class = KNeighborsClassifier(n_neighbors=5)
knn_class.fit(features_train, labels_train)
knn_class_pred = knn_class.predict(features_test)

# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(labels_train, knn_class.predict(features_train)))

# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(labels_test, knn_class_pred))

# Classification report
print("Classification report")
print(classification_report(labels_test, knn_class_pred))

print("---------------------------------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------------------------------")
print(" #### Klasifikacija na osnovu naslova i sadrzaja vesti #### ")
# forming train and test data
y = df['Class']
X = df['Title_and_Content_Parsed']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=8)

# Text representation: TF-IDF Vectors
# Parameter election
ngram_range = (1,2) # to consider both unigrams and bigrams
min_df = 10 # when building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold
max_df = 0.8 # when building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold 1
max_features =  300 # build a vocabulary that only consider the top max_features ordered by term frequency across the corpus

tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)

# fit and transform train features
features_train = tfidf.fit_transform(X_train).toarray()
labels_train = y_train
labels_train = y_train.astype(int)

# only transform test features
features_test = tfidf.transform(X_test).toarray()
labels_test = y_test
labels_test = y_test.astype(int)

# ** Multinomial Naïve Bayes **
print(" * * Multinomial Naïve Bayes * * ")
mnbc = MultinomialNB()
mnbc.fit(features_train, labels_train)
mnbc_pred = mnbc.predict(features_test)

# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(labels_train, mnbc.predict(features_train)))

# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(labels_test, mnbc_pred))

# Classification report
print("Classification report")
print(classification_report(labels_test,mnbc_pred))

# ** Gaussian Naïve Bayes **
print(" * * Gaussian Naïve Bayes * * ")
gnb = GaussianNB()
gnb.fit(features_train, labels_train)
gnb_pred = gnb.predict(features_test)

# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(labels_train, gnb.predict(features_train)))

# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(labels_test, gnb_pred))

# Classification report
print("Classification report")
print(classification_report(labels_test,gnb_pred))

# ** Support Vector Machine **
print(" * * Support Vector Machine * * ")
svm_class = svm.SVC(kernel="linear")
svm_class.fit(features_train, labels_train)
svm_class_pred = svm_class.predict(features_test)

# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(labels_train, svm_class.predict(features_train)))

# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(labels_test, svm_class_pred))

# Classification report
print("Classification report")
print(classification_report(labels_test, svm_class_pred))

# ** Random Forest **
print(" * * Random Forest * * ")
rf = RandomForestClassifier(random_state=0, n_estimators=100, bootstrap=True, max_features="sqrt")
rf.fit(features_train, labels_train)
rf_pred = rf.predict(features_test)

# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(labels_train, rf.predict(features_train)))

# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(labels_test, rf_pred))

# Classification report
print("Classification report")
print(classification_report(labels_test, rf_pred))

# ** K Nearest Neighbors **
print(" * * K Nearest Neighbors * * ")
knn_class = KNeighborsClassifier(n_neighbors=5)
knn_class.fit(features_train, labels_train)
knn_class_pred = knn_class.predict(features_test)

# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(labels_train, knn_class.predict(features_train)))

# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(labels_test, knn_class_pred))

# Classification report
print("Classification report")
print(classification_report(labels_test, knn_class_pred))

# Confusion matrix
aux_df = df[['Category', 'Class']].drop_duplicates().sort_values('Class')
conf_matrix = confusion_matrix(labels_test, knn_class_pred)
print("Confusion matrix: ")
print(conf_matrix)
plt.figure(figsize=(10,5))
sns.heatmap(conf_matrix,
            annot=True,
            xticklabels=aux_df['Category'].values,
            yticklabels=aux_df['Category'].values,
            cmap="Blues")
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.title('Confusion matrix')
plt.show()

print("---------------------------------------------------------------------------------------------------------------")
