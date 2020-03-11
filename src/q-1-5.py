import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pdb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys
from random import randrange

def load_data (data,folds=10):
    data = pd.read_csv(data)
    data = data.values
    n = data.shape[0]
    data = data[:,1:]
    # For performing Leave One Out Cross Validation
    # folds = n 
    dataset_split = np.array(cross_validation_split(data,folds))
    return dataset_split,folds

def cross_validation_split(dataset, folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

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
   
    for j in range(0,iters):
        params[0] = params[0] - (lr/X.shape[0]) * (sum(h - y) + del_penalty(lambda_value,params[0]))
        for k in range(1,X.shape[1]+1):
            params[k] = params[k] - (lr/X.shape[0]) * (sum((h-y) * X.T[k-1]) + del_penalty(lambda_value,params[k]))
        h = prediction(X,params)
        cost[j] = (1/X.shape[0]) * 0.5 * sum(np.square(h - y))
    train_error = sum(np.square(h - y))/X.shape[0]
    h_val = prediction(x_val,params)
    val_error= sum(np.square(h_val - y_val))/x_val.shape[0]  

    return cost, train_error, val_error, params

def get_train_val(dataset_split,k,folds):
    temp = np.concatenate((dataset_split[:k],dataset_split[k+1:]),axis=0)
    trainData = []
    for i in range(folds-1):
        if (i==0):
            trainData = temp[0]
        else:
            trainData = np.concatenate((trainData,temp[i]),axis=0)
    valData = dataset_split[k]
    return trainData,valData


if __name__ == "__main__":

    data = '../Dataset/data.csv'
    dataset_split,folds = load_data(data)
    lr = 0.0001
    iters = 300
    lambdaa = 1000001
    step = 100000
    lambda_value = [x for x in range(0,lambdaa,step)]
    train_error = np.ones(folds)
    val_error = np.ones(folds)
    cost = np.ones((folds,iters))
    params_k = np.ones((folds,8))
    params_all = np.ones((len(lambda_value),8))
    train_error_avg = np.ones(len(lambda_value))
    val_error_avg = np.ones(len(lambda_value))
    n_iterations = [x for x in range(iters)]

    for i in range(0,len(lambda_value)):
        for k in range (folds):
            trainData,valData = get_train_val(dataset_split,k,folds)
            x_train = trainData[:,:-1]
            x_train = normalization(x_train)
            y_train = trainData[:,-1]
            x_val = valData[:,:-1]
            if (x_val.shape[0]!=1):
                x_val = normalization(x_val)
            y_val = valData[:,-1]

            cost[k,:], train_error[k], val_error[k], params_k[k,:] = regression(x_train,y_train,x_val,y_val,lr,iters,lambda_value[i])
            if (k==0):
                plt.figure(1)
                plt.subplot(len(lambda_value)//3 + 1,3,i+1)
                plt.plot(n_iterations, cost[0])
                plt.xlabel('No. of iterations')
                plt.ylabel('Cost')
                plt.title('Cost Vs No. of iters for \u03BB = {}'.format(lambda_value[i]))
            plt.subplots_adjust(hspace=1.5)
            plt.subplots_adjust(wspace=0.3)
            plt.suptitle('Visualization of Cost Over Various \u03BB to Check Gradient Descent using {}-Fold Cross Validation'.format(folds))
        train_error_avg[i] = np.mean(train_error)
        val_error_avg[i] = np.mean(val_error)
        params_all[i] = np.mean(params_k,axis=0)
        print("Iterating over \u03BB values: {}/{}".format(i, len(lambda_value)-1), end='\r') 
    
    min_index = np.argmin(val_error_avg)
    print('The Minimun error is obtained for \u03BB = {}'.format(lambda_value[min_index]))

    lambda_value[1:] = np.log(lambda_value[1:])

    plt.figure(2)
    plt.plot(lambda_value, train_error_avg,label='Training')
    plt.plot(lambda_value, val_error_avg, label='Validation')
    plt.xlabel('$log(\lambda)$')
    plt.ylabel('Error')
    plt.title("Variation of Error with respect to $\lambda$ ($\lambda \in range(0,{},{})$ for 'Ridge' Regulrization using {}-Fold Cross Validation".format(lambdaa-1,step,folds))
    plt.legend()

    plt.figure(3)
    for i in range(params_all.shape[1]):
        plt.plot(lambda_value, params_all[:,i],label='Theta_%s'%i)
    plt.xlabel('$log(\lambda)$')
    plt.ylabel('Parameter')
    plt.title("Variation of Parameters with respect to $\lambda$ ($\lambda \in range(0,{},{})$ for 'Ridge' Regulrization using {}-Fold Cross Validation".format(lambdaa-1,step,folds))
    plt.legend()

    plt.show()