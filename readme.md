<details>
  <summary>Table Of Contents</summary>
  

  1. [About The Project](#about-the-project) 
        - [Built With](#built-with)

  2. [Getting Started](#getting-started)
  3. [Preprocess Data](#preprocess-data)
        - [Remove Punctuations and Special Characters](#remove-punc-and-sc)
        - [Remove Stopwords, Tokenize, Stem and Lemmatize text](#tokenize-stem-lemma-text)
        - [Labelling Data Using NLTK's Vader Lexicon](#labelling-data)
        - [Train Test Split](#train-test-split)
        - [Text Vectorization](#text-vectorization)
4. [Build and Compare Models](#build-and-compare-models)
</details>

# About The Project 
The purpose of this project is to scrape comments off youtube videos to perform sentiment analysis on these comments. This could help understand the audience feedback on specific videos and improve contents of youtube videos and etc.

### Built With
<img src="images_for_readme/image-1.png" width="200"><br>
<img src="images_for_readme/image.png" width="200"><br>
<img src="images_for_readme/image-2.png" width="200"><br>
<img src="images_for_readme/image-3.png" width="200">

# Getting Started
Before starting on the training of the model for sentiment analysis, comment_scraper class can be called and used together with the youtube video ID to scrape all the first level comments from the comments section.
```py
from comment_scraper import comment_scraper
comment_scraper = comment_scraper(videoId = "jb_lnAvZSa4")
```

# Preprocess Data
Before data can be passed into the model, preprocessing has to be done to ensure that the text data is in the correct format for the model to make predictions on.

## Remove Punctuations and Special Characters
The first step of preprocessing data would be to remove punctuations and special characters. This is to remove noise to make it easier for model to capture meaningful words and phrases.
```py
def preprocess(text):
    text = text.lower() 
    text=text.strip()  
    text=re.compile('<.*?>').sub('', text) 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text) 
    return text
df["text"] = df["text"].apply(lambda x: preprocess(x))
```
## Remove Stopwords, Tokenize, Stem and Lemmatize text
The second step is to remove stopwords. The purpose of removing stopwords is to remove any characters that do not add value to the model's predictions so that the model can focus on the important words. Afterwards, tokenization of text is done to splitting a phrase into separate words. Stemming is done to group words with similar meanings or roots together. Lastly, lemmatization is done to reduce a word to its root form.
```py
stop_words = set(stopwords.words("english"))
def remove_stopwords(text):
    stemmer = PorterStemmer()
    tokens = text.split()
    tokens = [word for word in tokens if word.lower() not in stop_words]
    stemmed_text = [stemmer.stem(token) for token in tokens]
    lemmatized_text = [stemmer.stem(token) for token in stemmed_text]
    return lemmatized_text
df["text"] = df["text"].apply(lambda x: remove_stopwords(x))
```
## Labelling Data using NLTK's Vader Lexicon
From the data that has been scraped using our scraper class that can be instantiated, it will only contain the texts available in the comment but not the sentiment of the comment. Therefore, it is important to determine the sentiment of each comment scraped. We will therefore be using NLTK's Vader Lexicon which is a rule-based sentiment analyzer to categorize the comments into positive or negative sentiments.
```py
sia = SIA()
def get_sentiment(tokens):
    text = " ".join(tokens)
    scores = sia.polarity_scores(text)
    if scores["compound"] >= 0.05:
        return "Positive"
    elif scores["compound"] <= 0.05:
        return "Negative"
    else:
        return "Neutral"
df["sentiment"] = df["text"].apply(lambda x: get_sentiment(x))
```

## Train Test Split
In order to train the model, it is important to split the data into training and testing data. The purpose of the training data is to train the model to make accurate predictions on unseen data while the purpose of the testing data is to test how accurate the model is in predicting unseen data.
```py
X = df["text"]
y = df["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
```

## Text Vectorization
Models cannot make predictions on data that are non numerical. Therefore, text have to be converted into numerical vectors before being able it can be fed into the model for predictions. In this case, we will be using TF-IDF vectorizer into numerical vectors which will then be fit into the model.
```py
tfidf_vectorizer = TfidfVectorizer()
X_train = tfidf_vectorizer.fit_transform(X_train)
X_test = tfidf_vectorizer.transform(X_test)
```

# Build And Compare Models
Last step is to build and compare models for this classification task of classifying comments into their respective sentiments of positive and negative represented by 0s and 1s. A wide variety of models are going to be tested to see which model yields the best results. \
| Model Type | Base Accuracy | Tuned Accuracy |
| :---: | :---: | :---: |
|Multinomial NB (Baseline Model)| 0.79 | NIL |
| Logistic Regression Model | 0.84 | 0.88 |
| Decision Tree Classifier | 0.85 | 0.85 |
| Random Forest Classifier | 0.86 | 0.74 |
| Xgboost Classifier | 0.86 | 0.87 |
| Support Vector Classifier | 0.84 | 0.86 |
| Bert Sequence Classification | 0.89 | NIL |

We can see from the above scores that the model with the most accurate predictions is the Bert Model with an accuracy of 89 percent on unseen data.





