# importing required modules
import csv
import re
from flask import Flask,request,render_template
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2		
from sklearn import svm    			
from sklearn.model_selection import train_test_split        
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
#functions
app = Flask(__name__) 
def read():
    x,y=[],[]
    with open('doc1.csv', 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            x.append(row[2])
            y.append(row[1])
    csvFile.close()
    return x,y 

# it gives pos tag for each word
def lemmatize_verbs(x):
    word_tokens = word_tokenize(x)
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for w in word_tokens:
        lemma = lemmatizer.lemmatize(w, pos='v')
        lemmas.append(lemma)
    return ' '.join(lemmas)

# it removes stop words like the, a, is, that etc
def stopword(f):
    stop_words = set(stopwords.words('english'))
    filtered_sentence=[]
    for i in f:
        word_tokens = word_tokenize(i)
        filtered_sentence.append(' '.join([w for w in word_tokens if not w in stop_words]))
    return filtered_sentence

# applies preprocessing techniques like removal of punctuation marks, single letters, digits 
def preprocessing(t):
    l=[]
    for i in t:
        i=i.lower() 
        i = re.sub(r'\W', ' ', i)
        i=re.sub(r'\s+[a-zA-Z]\s+', ' ', i) 
        i=re.sub(r'\s+', ' ', i, flags=re.I)
        l.append(i)
    return l

# implementes tfi-idf statistic
def tfi_idf():
    vectorizer = TfidfVectorizer ()
    processed_features = vectorizer.fit_transform(clean_text).toarray()
    return processed_features,vectorize

# svm algo
def supportvec():
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred,clf 
# logistic regression algo
def logreg():
    logreg = LogisticRegression(solver='lbfgs')
    logreg.fit(X_train,y_train)
    y_pred=logreg.predict(X_test)
    return y_pred,logreg

# combining both logistic regression and svm using ensemble methods
def ensemble():
    model1 = n2
    model2 = n
    model = VotingClassifier(estimators=[('lr', model1), ('svm', model2)])
    model.fit(X_train,y_train)
    print(model.score(X_test,y_test))
    
    return model


def trim(t):
    li=[]
    for i in t:
        i=re.sub(r'#[a-zA-Z]+','',i)
        i=re.sub(r'&quot;','',i)
        i=re.sub(r'@\w+','',i)
        i=re.sub(r'https?://+S','',i)
        li.append(lemmatize_verbs(i))
    return li

# returns predicted value
def predict_sent(c,sen):
    y_pred = c.predict(sen)
    #print("Prediction : ",pre)
    return y_pred

# renders home page of web app
@app.route('/') 
def home():  
   return render_template('home.html')

# retrievs the comment from web app and renders predicted value in web page
@app.route('/success',methods = ['POST'])  
def print_data(): 
    msg=request.form['message']
    t1=preprocessing([msg])
    t2=stopword(t1)
    t3=trim(t2)
    t4=vect.transform(t3).toarray()
    #t5=fix.transform(t4)
    res=predict_sent(n3,t4)
    return render_template('result.html',prediction = res)


if __name__=='__main__':
    data,label=read()
    removestop=stopword(data)
    trimout=trim(removestop)
    clean_text=preprocessing(trimout)
    tfi_idf_model,vect=tfi_idf()
    tar=np.asarray(label)
    tar=tar.astype('int32')   
    X_train, X_test, y_train, y_test = train_test_split(tfi_idf_model, tar, test_size=0.2,random_state=999)
    svm_target,n=supportvec() 
    logreg_target,n2=logreg()
    n3=ensemble()
    app.run()

   


    
    
    
    
    
