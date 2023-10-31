#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import uvicorn
import scipy.sparse
from fastapi import FastAPI as fapi
from resume import resume
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import pickle

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
app = fapi()


with open('nb_model-2.pkl', 'rb') as model_file:
    nb_model = pickle.load(model_file)
with open('count_vect.pkl', 'rb') as count_vect_file:
    count_vect = pickle.load(count_vect_file)

df = pd.read_csv('UpdatedResumeDataSet.csv', encoding='ISO-8859-1')


# In[10]:


@app.get('/')
def index():
    return{'message': 'Hello'}


# In[11]:


@app.get('/{Name}')
def get_name(name:str):
    return {'message':'hello,{name}'  }


# In[18]:


from fastapi import FastAPI, File, UploadFile
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import io
import math

app = FastAPI()

@app.post('/texttopdf')



def convertPDFtoText(pdf_file: UploadFile):
    output_string = io.StringIO()
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, output_string, laparams=laparams)
    
    with open(path, 'rb') as fp:
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.get_pages(fp):
            interpreter.process_page(page)
    
    text = output_string.getvalue()
    
    device.close()
    output_string.close()
    
    return text

# Define your text summarization logic here
@app.post('/summarise')
def summarize(text, num_sentences=10):
  
  
  
  words = word_tokenize(text.lower())
  

  
  
  # Remove stopwords
  words = [w for w in words]
  
  # Extract skills and experience keywords
  skills = re.findall(r'(\w*?(?:ing|ed)|skills|experience)', ' '.join(words))
  keys = set(skills)
  
  # Calculate word frequency  
  freq = FreqDist(words)
  
  # Score sentences
  scores = {}
  for i, sent in enumerate(text.split('.')):
    score = 0
    for word in word_tokenize(sent.lower()):
      if word in keys:
        score += 2
      elif word in freq:
        score += freq[word]
        
    scores[i] = score
    
  # Sort sentences by score  
  ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
  
  # Return top sentences
  ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
  indices = [i for i, s in ranked]
  top_idx = indices[:int(num_sentences)]
  
  
  summarized = '. '.join([text.split('.')[idx] for idx in top_idx])
  return summarized

@app.post('/predict_job_category')

def predict_job_category(text:str):
    
    tfidf_transformer = TfidfTransformer()
    # Create a pipeline with CountVectorizer and Multinomial Naive Bayes
    text_clf = Pipeline([('vect', count_vect), ('tfidf', tfidf_transformer), ('clf', nb_model)])
    
    
    X_train = df['Resume']
    y_train = df['Category']  

    # Fit the model
    text_clf.fit(X_train, y_train)
    
    # Predict the job category based on the input text
    predicted_category = text_clf.predict([text])[0]
    
    return {"category": predicted_category}

# In[6]:





# In[ ]:




