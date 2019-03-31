# coding: utf-8

# In[1]:


import sys
import csv
import numpy as np


# In[ ]:


files = list(sys.argv)

train_input = files[1]
test_input = files[2]
train_out = files[3]
test_out = files[4]
metrics_out = files[5]

num_epoch = int(files[6]) #number of epochs
hidden_units = int(files[7]) #number of hidden units
init_flag = int(files[8]) #choose between 1 random or 2 zero
learning_rate = float(files[9])


# In[2]:





# In[2]:


#activation of the first layer
def sigmoid(a):
    return 1/(1+np.exp(-a))


# In[3]:


def NNforward(train,train_y,alpha,beta):
    a = np.dot(alpha,train.T)
    z = sigmoid(a)
    z= np.insert(z,0,1)
    
    b = np.dot(beta,z)
    exp_scores = np.exp(b.T)
    probs = exp_scores / np.sum(exp_scores) #softmax
    y_hat = np.argmax(probs)
    return probs,y_hat,z


# In[4]:


def NNBackward(train,train_y,alpha,beta,probs,z):
    dLdB = np.copy(probs)
    dLdB[train_y] = dLdB[train_y] -1

    dLdB = np.reshape(dLdB, (-1, len(dLdB)))
    z = np.reshape(z, (-1, len(z)))
    dLdBeta = np.dot(dLdB.T, z)
    
    beta_star = np.copy(beta[:,1:])
    #print(beta_star)
    dLdZ = np.dot(dLdB,beta_star)
    z_star = np.copy(z[0][1:])
    
    #print(z.shape,dLdZ.shape)
    dLdA = dLdZ*z_star*(1-z_star)
    #print(dLdA.shape)
    train = np.reshape(train, (-1, len(train)))
    dLdAlpha = np.dot(dLdA.T,train)
    return dLdAlpha, dLdBeta


# In[5]:


def entropy_loss(probs,y):
    l = probs[y] #find the probability
    s = -np.log(l)   
    return s


# In[6]:


#hidden_units = 4 #number of hidden units
#init_flag = 2 #choose between 1 random or 2 zero
#num_epoch = 2 #number of epochs
#learning_rate = 0.1

#train_input = "./handout/smallTrain.csv"
#test_input = "./handout/smallTest.csv"
#train_out = "trainOut.labels"
#test_out = "testOut.labels"
#metrics_out = "metrics.txt"


# In[7]:


train_file = [[int(code) for code in line.split(',')] for line in open(train_input,'r').read().splitlines()]
test_file = [[int(code) for code in line.split(',')] for line in open(test_input,'r').read().splitlines()]

train = np.array(train_file)
test = np.array(test_file)
train_y = np.copy(train[:,0])
test_y = np.copy(test[:,0])

train[:, 0] =  1  #initialize bias term, x0=1 is fixed
test[:, 0] =  1  #initialize bias term 


if (init_flag==1): #random
    alpha = np.random.random_sample((hidden_units,len(train[0])))/5 - 0.1
else:
    alpha = np.zeros((hidden_units,len(train[0])))
beta = np.zeros((10,hidden_units+1)) 


# In[8]:


file_metrics = open(metrics_out,'w')
train_out = open(train_out,'w')
test_out = open(test_out,'w')


# In[9]:


for epoch in range(num_epoch):
    #print(alpha)
    entropy = {}
    #print(epoch)
    
    for i in range(len(train)):
        x = train[i]
        y = train_y[i]    
        probs,y_hat,z = NNforward(x,y,alpha,beta)
        dLdAlpha, dLdBeta = NNBackward(x,y,alpha,beta,probs,z)
        beta = beta - dLdBeta * learning_rate
        alpha = alpha - dLdAlpha * learning_rate
    
    ent = 0
    error_train = 0
    #print(beta)
    #print(alpha)
    for i in range(len(train)):
        x = train[i]
        y = train_y[i]
        probs,y_hat,z = NNforward(x,y,alpha,beta)
        ent = entropy_loss(probs,y) + ent
        if (epoch==num_epoch-1):
            train_out.write(str(y_hat)+"\n")
        if (y_hat != y):
            error_train = error_train +1
    entropy[str(epoch)+"train"] = ent/len(train)
    print("epoch="+str(epoch+1) + " crossentropy(train): " + "{:.11f}".format(ent/len(train)))
    file_metrics.write("epoch="+str(epoch+1) + " crossentropy(train): " + "{:.11f}".format(ent/len(train))+"\n")
    
    error_test= 0
    ent_test = 0
    for i in range(len(test)):
        x = test[i]
        y = test_y[i]
        probs,y_hat,z = NNforward(x,y,alpha,beta)
        ent_test = entropy_loss(probs,y) + ent_test
        if (epoch==num_epoch-1):
            test_out.write(str(y_hat)+"\n")
        #print(y_hat)
        if (y_hat != y):
            error_test = error_test +1
    entropy[str(epoch)+"test"] = ent_test/len(test)
    #print("\nepoch="+str(epoch+1) + " crossentropy(test): " + str(ent_test/len(test)))
    file_metrics.write("epoch="+str(epoch+1) + " crossentropy(test): " + "{:.11f}".format(ent_test/len(test))+"\n")
#print("error(train):" +str(error_train/len(train)))
#print("error(test):" +str(error_test/len(test)))
file_metrics.write("error(train): " +str(error_train/len(train))+"\n")
file_metrics.write("error(test): " +str(error_test/len(test)))


# In[9]:


file_metrics.close()
train_out.close()
test_out.close()


# In[ ]:




