#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd

Foods= pd.read_csv("Food.csv")


# In[27]:


Foods.head()


# In[28]:


import re

def clean_title(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title
Foods["clean_title"] = Foods["title"].apply(clean_title)
Foods


# In[29]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2))

tfidf = vectorizer.fit_transform(Foods["clean_title"])
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -1)[-1:]
    results = Foods.iloc[indices].iloc[::-1]
    
    return results


# In[30]:


import ipywidgets as widgets
from IPython.display import display

Foods_input = widgets.Text(
    value='',
    description='Your Name:',
    disabled=False
)
Foods_list = widgets.Output()

def on_type(data):
    with Foods_list:
        Foods_list.clear_output()
        title = data["new"]
        if len(title) > 1:
            display(search(title))

Foods_input.observe(on_type, names='value')


display(Foods_input, Foods_list)

