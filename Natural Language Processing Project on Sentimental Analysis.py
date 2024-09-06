#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re


# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Wordcloud
from wordcloud import wordcloud


nltk.download('stopwords')
nltk.download('wordnet')

# Sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Evaluation Metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[2]:


from scikitplot.metrics import plot_confusion_matrix


# In[3]:


df = pd.read_csv('education.csv')


# In[4]:


df


# In[5]:


df.reset_index(inplace=True, drop=True)


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.shape


# In[9]:


df.sample(5)


# In[10]:


df.head(2)


# In[11]:


df['Label'].unique()


# In[12]:


df['Label'].value_counts()


# In[13]:


df['Label'].value_counts().plot.bar()


# * Lemmatizer: https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/

# In[14]:


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

lm = WordNetLemmatizer()


# In[15]:


def transformation(df_column):
    output = []
    for i in df_column:
        new_text = re.sub('[^a-zA-Z]', ' ', str(i))
        new_text = new_text.lower()
        new_text = new_text.split()
        new_text = [lm.lemmatize(j) for j in new_text if j not in set(stopwords.words('english'))]
        output.append(' '.join(str(k) for k in new_text))
  
    return output


# In[16]:


var = transformation(df['Text'])


# In[17]:


var


# In[18]:


from wordcloud import WordCloud


# In[19]:


# Word Cloud
plt.figure(figsize=(50,28))
word = ''
for i in var:
  for j in i:
    word += " ".join(j)

wc = WordCloud(width=1000, height= 500, background_color='white', min_font_size=10).generate(word)
plt.imshow(wc)


# In[20]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(1, 2))
traindata = cv.fit_transform(var)
X_train = traindata
y_train = df['Label']


# In[21]:


X_train


# In[22]:


model = RandomForestClassifier()


# In[23]:


# Hyper Parameter Tuning

parameters = {'max_features':('auto', 'sqrt'),
              'n_estimators': [500, 1000, 1500],
              'max_depth': [5,10, None],
              'min_samples_leaf':[1, 2, 5, 10],
              'min_samples_split':[5, 10, 15],
              'bootstrap':[True, False]}


# In[24]:


parameters


# In[25]:


grid_search = GridSearchCV(model, 
                           parameters, 
                           cv=5,
                           return_train_score = True,
                           n_jobs=1)


# In[ ]:


grid_search.fit(X_train, y_train)


# In[ ]:


grid_search.best_params_


# In[ ]:


rfc = RandomForestClassifier(max_features= grid_search.best_params_['max_features'],
                             n_estimators= grid_search.best_params_['n_estimators'],
                             max_depth= grid_search.best_params_['max_depth'],
                             min_samples_leaf= grid_search.best_params_['min_samples_leaf'],
                             min_samples_split= grid_search.best_params_['min_samples_split'],
                             bootstrap= grid_search.best_params_['bootstrap'])


# In[ ]:


rfc.fit(X_train, y_train)


# In[ ]:


test_data = 
X_test, y_test


# In[ ]:


y_pred = rfc.predict(X_test)


# In[ ]:


# Model Evaluation
accuracy_score(y_test, y_pred)


# In[ ]:


def sentimental_analysis(input):
  new_input = tranformation(input)
  transformed_input = cv.transform(new_input)
  prediction = rfc.predict(transformed_input)
  if prediction == 0:
    print('Negative Sentiment')
  elif prediction == 1:
    print('Positive Sentiment')
  else:
    print('Invalid Sentiment')


# In[ ]:


input = "Today I was playing in the park and I fell"
inp = input("")


# In[ ]:


sentimental_analysis(inp)


# In[ ]:




