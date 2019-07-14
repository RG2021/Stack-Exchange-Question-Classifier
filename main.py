# Enter your code here. Read input from STDIN. Print output to STDOUT
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing

i=0
train=[]
topic=[]
with open('training.json','r') as f:
    for line in f:
        i=i+1
        if(i==1):
            continue
        j = json.loads(line)
        topic.append(j["topic"])
        train.append(j["excerpt"])

train=np.array(train)
test=[]
N = int(input())
for i in range(N):
    j = json.loads(input())
    test.append(j['excerpt'])
test=np.array(test)
    
vect = TfidfVectorizer(sublinear_tf=True, analyzer = 'word', ngram_range=(1,1), stop_words='english', max_features=None)
X = vect.fit_transform(train)
le=preprocessing.LabelEncoder()
Y=le.fit_transform(topic)
X_test = vect.transform(test)

clf = MultinomialNB()
clf.fit(X, Y)

predictions = clf.predict(X_test)
for i in predictions:
    a=le.inverse_transform([i])
    print(a[0])
