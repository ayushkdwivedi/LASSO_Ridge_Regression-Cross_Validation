import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pdb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys

def load_data (data):
    data = pd.read_csv(data)
    columns = data.columns
    data = data.values
    # print(data.shape)
    # data = data.iloc[:20,:]
    trainData, valData = train_test_split(data, test_size = 0.2, random_state = 42)

    y_val = valData[:,-1]
    x_val = valData[:,1:-1]

    y_train = trainData[:,-1]
    x_train = trainData[:,1:-1]
   

    return x_val,y_val,x_train,y_train,columns

def normalization(X):
    mu = np.ones(X.shape[1])
    std = mu

    for i in range(0, X.shape[1]):
        mu[i] = np.mean(X.T[i])
        std[i] = np.std(X.T[i])
        for j in range(0, X.shape[0]):
            X[j][i] = (X[j][i] - mu[i])/std[i]
    return X

def prediction(x_test,params):
    x = x_test
    temp = np.ones((x_test.shape[0],1))
    x=np.concatenate((temp,x),axis=1)
    y_pred = np.matmul(x,params)
    return y_pred

def del_penalty(lam,para):
    return lam*para

def regression(X, y, x_val,y_val, lr, iters, lambda_value):

    params = np.zeros(X.shape[1]+1)
    h = prediction(X, params)

    cost = np.ones(iters)
    train_error = np.ones(len(lambda_value))
    val_error = np.ones(len(lambda_value))
    params_all = np.ones((len(lambda_value),len(params)))

    for i in range(0,len(lambda_value)):
        for j in range(0,iters):
            params[0] = params[0] - (lr/X.shape[0]) * (sum(h - y) + del_penalty(lambda_value[i],params[0]))
            for k in range(1,X.shape[1]+1):
                params[k] = params[k] - (lr/X.shape[0]) * (sum((h-y) * X.T[k-1]) + del_penalty(lambda_value[i],params[k]))
            h = prediction(X,params)
            if (i==0):
                cost[j] = (1/X.shape[0]) * 0.5 * sum(np.square(h - y))
        train_error[i] = sum(np.square(h - y))/X.shape[0]
        h_val = prediction(x_val,params)
        val_error[i] = sum(np.square(h_val - y_val))/x_val.shape[0]
        params_all[i,:] = params
        print("Iterating over \u03BB values: {}/{}".format(i, len(lambda_value)), end='\r') 

    params_all = params_all.reshape(len(lambda_value),len(params))
    return params_all, cost, train_error, val_error

if __name__ == "__main__":

    data = '../Dataset/data.csv'

    x_val,y_val,x_train,y_train,columns = load_data(data)
    x_train = normalization(x_train)
    x_val = normalization(x_val)
    
    i = 0
    plt.figure(1)
    for col in range(x_train.shape[1]):
        i+=1
        plt.subplot(4,2,i)
        plt.scatter(x_train[:,col],y_train)
        plt.title("Plot of 'Admission Probability' vs '%s'" %columns[col+1])
        plt.xlabel('%s'%columns[col+1])
        plt.ylabel('Adm Prob') 
    plt.subplots_adjust(hspace=1.5)
    plt.suptitle('Dataset Visualization')
    
    lr = 0.0001
    iters = 300
    lambdaa = 1001
    step = 10
    lambda_value = [x for x in range(0,lambdaa,step)]
    params_all, cost, train_error, val_error = regression(x_train,y_train,x_val,y_val,lr,iters,lambda_value)
    print(params_all)
    
    cost = list(cost)
    n_iterations = [x for x in range(iters)]
    lambda_value[1:] = np.log10(lambda_value[1:])
    plt.figure(2)
    plt.plot(n_iterations, cost)
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost')
    plt.title('Variation of cost with respect to number of iterations')

    plt.figure(3)
    plt.plot(lambda_value, train_error,label='Training')
    plt.plot(lambda_value, val_error, label='Validation')
    plt.xlabel('$log(\lambda)$')
    plt.ylabel('Error')
    plt.title("Variation of Error with respect to $\lambda$ ($\lambda \in range(0,{},{})$ for 'Ridge' Regulrization".format(lambdaa-1,step))
    plt.legend()

    plt.figure(4)
    for i in range(params_all.shape[1]):
        plt.plot(lambda_value, params_all[:,i],label='Theta_%s'%i)
    plt.xlabel('$log(\lambda)$')
    plt.ylabel('Parameter')
    plt.title("Variation of Parameters with respect to $\lambda$ ($\lambda \in range(0,{},{})$ for 'Ridge' Regulrization".format(lambdaa-1,step))
    plt.legend()

    plt.show()