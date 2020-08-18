#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import string
import pickle


# In[2]:


stop = set(stopwords.words('english'))
good_tokens = {}
good_tokens_ham = {}
good_tokens_spam = {} 


# In[4]:


#Specify Number of Training Sample of Spam and Ham
ham_no = 8000
spam_no = 6000


# In[5]:


ham = {}
spam = {}


# In[6]:


table = str.maketrans('', '', string.punctuation)
for i in range(1,ham_no+1):
    filename = './MyData/TrainHam/email'+str(i)+'.txt'
    text = open(filename,encoding='latin-1').read()
    tokens = word_tokenize(text)
    tokens = tokens[1:]
    temp = [w for w in tokens if not w in stop and len(w)>1]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in temp]
    words = [''.join([w.translate(table) for w in word]) for word in words]
    words = [word for word in words if (any(l.isdigit() for l in word)==False)]
    words = [w for w in words if not w in stop and len(w)>1]
    ham[i] = words
    for val in words:
        if val in good_tokens:
            good_tokens[val] += 1
        else:
            good_tokens[val] = 1

for i in range(1,spam_no+1):
    filename = './MyData/TrainSpam/email'+str(i)+'.txt'
    text = open(filename,encoding='latin-1').read()
    tokens = word_tokenize(text)
    tokens = tokens[1:]
    temp = [w for w in tokens if not w in stop and len(w)>1]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in temp]
    words = [''.join([w.translate(table) for w in word]) for word in words]
    words = [word for word in words if (any(l.isdigit() for l in word)==False)]
    words = [w for w in words if not w in stop and len(w)>1]
    spam[i] = words
    for val in words:
        if val in good_tokens:
            good_tokens[val] += 1
        else:
            good_tokens[val] = 1
            


# In[7]:


len(good_tokens)


# In[34]:


dict_words = sorted(good_tokens.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)[:30000]


# In[38]:


model = {}
for word in dict_words:
    model[word[0]] = np.array([0,0])
print("done")
j=0
for word in dict_words:
    print(j)
    for i in range(1,ham_no+1):
        if word[0] in ham[i]:
            model[word[0]][0] += 1
    for i in range(1,spam_no+1):
        if word[0] in spam[i]:
            model[word[0]][1] += 1
    j+=1


# In[41]:


model2 = [model,ham_no,spam_no]
pickle_out = open("model2.pickle","wb")
pickle.dump(model2, pickle_out)
pickle_out.close()


# In[48]:


model2[0]


# In[ ]:





# In[ ]:




