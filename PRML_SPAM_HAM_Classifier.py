#!/usr/bin/env python
# coding: utf-8

# In[26]:


from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import string
import pickle


# In[27]:


stop = set(stopwords.words('english'))
table = str.maketrans('', '', string.punctuation)


# In[28]:


def process_mail(text):
    tokens = word_tokenize(text)
    tokens = tokens[:]
    temp = [w for w in tokens if not w in stop and len(w)>1]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in temp]
    words = [''.join([w.translate(table) for w in word]) for word in words]
    words = [word for word in words if (any(l.isdigit() for l in word)==False)]
    words = [w for w in words if not w in stop and len(w)>1]
    words = list(set(words))
    return words

def classify_mail(text,m):
    model = m[0]
    ham_no = m[1]
    spam_no = m[2]
    phi_j_y_1 = 1.0
    phi_j_y_0 = 1.0
    phi = spam_no/(float)(spam_no + ham_no)
    phi_o = 1-phi
    mail = list(set(text))
    for word in mail:
        if word in model:
            temp1 = (model[word][1]+1)/float(spam_no+2)
            temp2 = (model[word][0]+1)/float(ham_no+2)
#             if temp1>temp2:
#                 print("%s likely spam %f"%(word,temp1))
#             else:
#                 print("%s likely ham %f"%(word,temp2))
            phi_j_y_1 *= (model[word][1]+1)/float(spam_no+2)
            phi_j_y_0 *= (model[word][0]+1)/float(ham_no+2)
        else:
            phi_j_y_1 *= 1/float(spam_no+2)
            phi_j_y_0 *= 1/float(ham_no+2)
    phi_j_y_1 *= phi
    phi_j_y_0 *= phi_o
    norm = 1
    if phi_j_y_1==0 and phi_j_y_0==0:
        #print("Cannot figure out")
        return False
    is_spam = phi_j_y_1*norm/((phi_j_y_1+phi_j_y_0)*norm)
    is_ham = phi_j_y_0*norm/((phi_j_y_1+phi_j_y_0)*norm)
#     if is_spam > is_ham:
#         print("Mail is spam with likelihood %f"%is_spam)
#     else:
#         print("Mail is ham with likelihood %f"%is_ham)
    return is_spam>is_ham
        
    
# #         print(phi_j_y_1)
#         if word not in model:
#             phi_j_y_1 *= 0.5
#             phi_j_y_0 *= 0.5
#         else:
#             if word in good_tokens_spam:
#                 temp = good_tokens_spam[word]/float(spam_no)
#                 phi_j_y_1 *= temp
# #                 if phi_j_y_1 == 0:
# #                     print("Why zero in spam!")
# #                     print(word,good_tokens_spam[word])
#             else:
#                 temp = good_tokens_ham[word]/3672.0
# #                 print(phi_j_y_0)
#                 phi_j_y_0 *= temp
# #                 if phi_j_y_0 == 0:
# #                     print("Why zero in ham!")
# #                     print(word,good_tokens_ham[word])
# #     if phi_j_y_1 == 0 and phi_j_y_0==0:
# #         print(i)
#     is_spam = phi_j_y_1*phi #/ (phi_j_y_1*phi + phi_j_y_0*phi_o)
#     is_not_spam =  phi_j_y_0*phi_o #/ (phi_j_y_1*phi + phi_j_y_0*phi_o)
# #    print(is_spam,is_not_spam)
#     return is_spam > is_not_spam


# In[22]:


#Load Pickle File to acquire Model Info


# In[29]:


pickle_in = open("model2.pickle","rb")
m = pickle.load(pickle_in)


# In[32]:


m = [m[0][0],m[1],m[2]]


# In[41]:


ham_no = m[1]
spam_no = m[2]


# In[52]:


# Read files from test data to start predicting
count1 = 0
for i in range(1,1450):
    filename = './MyData/TestSpam/email'+str(i)+'.txt'
    text = open(filename,encoding='latin-1').read()
    mail = process_mail(text)
    if classify_mail(mail,m):
        count+=1
acc1 = count1*100/1450        


# In[53]:


count1


# In[47]:


print(acc1)


# In[54]:


count2 = 0
for i in range(1,1450):
    filename = './MyData/TestHam/email'+str(i)+'.txt'
    text = open(filename,encoding='latin-1').read()
    mail = process_mail(text)
    if classify_mail(mail,m)==0:
        count2+=1
acc2 = count2*100/1450        


# In[55]:


print(acc2,count2)


# In[ ]:




