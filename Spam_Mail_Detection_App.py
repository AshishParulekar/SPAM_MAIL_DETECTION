import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
import string
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
#nltk.download('wordnet')
import nltk
#dler = nltk.downloader.Downloader()
#dler._update_index()
#dler.download('all')
import streamlit as st
#from PIL import Image


#####  Saved_Model :-

    
saved_model=pickle.load(open('Spam_Mail_Detection','rb'))

Cv=saved_model[0]
Log=saved_model[1]
####   Functions : -

stop_words=set(stopwords.words('english'))
poctuation=string.punctuation

text='Congratulations! You’re being offered a no-interest Visa credit card. Click here to claim: https://bit.ly/07tjA786'
def Text_processing(text):
    process_text=[]
    tk=nltk.word_tokenize(text)
    stop_words=set(stopwords.words('english'))
    for i in tk:
        i=i.lower()
        if ((i not in stop_words) & (i not in string.punctuation)):
            la=WordNetLemmatizer()
            i=la.lemmatize(i)
            process_text.append(i)
    return  process_text

def Spam_Detection(text):
    text=Text_processing(text)
    text=" ".join(text)
    Vector=Cv.transform([text]).toarray()
    result=Log.predict(Vector)
    if(result==1):
        #image = Image.open('NO_SPAM.jpg')
        #st.image(image, caption='Sunrise by the mountains')
        st.image('https://raw.githubusercontent.com/AshishParulekar/SPAM_MAIL_DETECTION/main/NO_SPAM.png?token=GHSAT0AAAAAABXHPZKY5HF4AVBDZ2X42OVSYZVOBVQ')
        #return '** Spam **'
    else:
         st.image('https://raw.githubusercontent.com/AshishParulekar/SPAM_MAIL_DETECTION/main/NO_SPAM.png?token=GHSAT0AAAAAABXHPZKYBDQFMLLDUTOSAYFQYZVOVIA')
        #return '** Not_Spam **'


# Streamlit_Code********************

st.title('Spam Mail Detector')
C1,C2 = st.columns((8,2))
C1.write(' ')
C1.subheader('Please Enter Your Mail Message')
input_text=C1.text_input(' ')

Bu=C1.button('Check')

if (Bu==True):
    text=Spam_Detection(input_text)
    #C1.subheader('Your Mail or Messege is ',+text )


