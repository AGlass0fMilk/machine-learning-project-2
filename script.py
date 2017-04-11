import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix

    # 1.) Split data into k parts (number of classes)
    # First we must know how many classes we are dealing with...


    # Returns a list of unique elements in the y (classes) vector
    # This gives us a list of the classes (and how many, k, there are)
    # It also will return how many of each class occurred in the array
    # We can then get p(Y=y) using the counts (divide each count by the total sum)
    classes, counts = np.unique(y, return_counts=True)
    classes = np.array(classes).astype(int)
    counts = np.array(counts).astype(int)
    # N = np.shape(X)[0]
    d = np.shape(X)[1]
    k = np.size(classes)
    # p_of_y = counts / np.sum(counts)

    # Split the data into k-parts
    X_parts = []
    for my_class in classes:
        indices, garbage = np.where(y == my_class)  # This will return indices where the class is my_class
        X_parts.append(X[indices])  # Select X where class is y

    X_parts = np.array(X_parts)

    # Now the data is split
    # 2.) Do MLE and train the QDA
    means = np.zeros((k, d))
    for my_class in classes:
        # Compute means for each column for each class
        means[my_class - 1] = np.mean(X_parts[my_class - 1], axis=0)

    covmat = np.cov(X, rowvar=False)
    means = means.transpose()  # Make it d x k
    
    # IMPLEMENT THIS METHOD
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # N = 150
    # d = 2
    # k = 5

    # 1.) Split data into k parts (number of classes)
    # First we must know how many classes we are dealing with...


    # Returns a list of unique elements in the y (classes) vector
    # This gives us a list of the classes (and how many, k, there are)
    # It also will return how many of each class occurred in the array
    # We can then get p(Y=y) using the counts (divide each count by the total sum)
    classes, counts = np.unique(y, return_counts=True)
    classes = np.array(classes).astype(int)
    counts = np.array(counts).astype(int)
    #N = np.shape(X)[0]
    d = np.shape(X)[1]
    k = np.size(classes)
    #p_of_y = counts / np.sum(counts)

    # Split the data into k-parts
    X_parts = []
    for my_class in classes:
        indices, garbage = np.where(y == my_class) # This will return indices where the class is my_class
        X_parts.append(X[indices]) # Select X where class is y

    X_parts = np.array(X_parts)

    # Now the data is split
    # 2.) Do MLE and train the QDA
    means = np.zeros((k, d))
    covmats = np.zeros((k, d, d))
    for my_class in classes:
        # Compute means for each column for each class
        means[my_class-1] = np.mean(X_parts[my_class-1],axis=0)
        covmats[my_class-1] = np.cov(X_parts[my_class-1], rowvar=False)

    means = means.transpose() # Make it d x k

    # IMPLEMENT THIS METHOD
    return means, covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # 3.) Use trained QDA model to compute theta for p(y)

    # Untranspose means...
    means = means.transpose()

    # See qdaLearn above for explanation of this next chunk
    classes, counts = np.unique(ytest, return_counts=True)
    classes = np.array(classes).astype(int)
    counts = np.array(counts).astype(int)
    N = np.shape(Xtest)[0]
    d = np.shape(Xtest)[1]
    k = np.shape(means)[0]  # Can't get this from classes here, since the grid data class is all 0's
    p_of_y = counts / np.sum(counts)  # In the case of the grid data this should all give us 1
    if (np.shape(p_of_y) != (k,)):
        p_of_y = np.ones((1, k))
        classes = np.arange(1, k + 1)

    p_of_y = np.ones((1, k))

    # Now our "favorite expression" comes into play
    # To calculate p(X=x|Y=y)
    def calculate_pdf(D, Sigma, x, mu):

        x_minus_mu = np.reshape((x - mu), (np.shape(x)[0], 1))
        trans = np.transpose(x_minus_mu)  # .transpose() doesnt work on 1-D vectors...

        inside_exp = np.dot(np.dot(trans, inv(Sigma)), x_minus_mu)
        exp = np.exp(-0.5 * inside_exp)
        out_exp = 1 / sqrt(det(Sigma))
        return out_exp * exp

    # Each row represents a sample, each column is the class conditional probability for that sample
    p_of_x_given_y = np.zeros((N, k))

    # Final probability we care about
    p_of_y_given_x = np.zeros((N, k))

    ypred = np.zeros((N, 1))

    # Loop through all samples
    for i in range(0, N):
        sample = Xtest[i]
        # For each sample, calculate the probabilities of each class
        for my_class in classes:
            p_of_x_given_y[i, my_class - 1] = calculate_pdf(d, covmat, sample, means[my_class - 1])

        # Calculate p_of_y_given_x
        p_of_y_given_x[i] = (p_of_y * p_of_x_given_y[i]) / np.dot(p_of_y, p_of_x_given_y[i])

        # Get the maximum probability (add 1 since the classes aren't 0 indexed)
        ypred[i] = np.argmax(p_of_y_given_x[i]) + 1

    # Calculate accuracy
    acc = 0
    match, garbage = np.where(ypred == ytest)
    acc = np.shape(match)[0] / np.shape(ytest)[0]

    #compare = np.concatenate((ypred, ytest), axis = 1)
    #print(compare)

    # IMPLEMENT THIS METHOD
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # 3.) Use trained QDA model to compute theta for p(y)

    # Untranspose means...
    means = means.transpose()

    # See qdaLearn above for explanation of this next chunk
    #classes, counts = np.unique(ytest, return_counts=True)
    #classes = np.array(classes).astype(int)
    #counts = np.array(counts).astype(int)
    N = np.shape(Xtest)[0]
    d = np.shape(Xtest)[1]
    k = np.shape(means)[0] # Can't get this from classes here, since the grid data class is all 0's

    classes = np.arange(1, k+1)

    #p_of_y = np.ones((1, k))

    # Now our "favorite expression" comes into play
    # To calculate p(X=x|Y=y)
    def calculate_pdf(D, Sigma, x, mu):
        x_minus_mu = np.reshape((x - mu), (np.shape(x)[0], 1))
        trans = np.transpose(x_minus_mu)# .transpose() doesnt work on 1-D vectors...

        inside_exp = np.dot(np.dot(trans,inv(Sigma)), x_minus_mu)
        exp = np.exp(-0.5 * inside_exp)
        #out_exp = 1/(((2*np.pi)**(D/2))*sqrt(det(Sigma)))
        #out_exp = 1/(sqrt(((2*np.pi)**D)*det(Sigma)))
        out_exp = 1 / sqrt(det(Sigma))
        return out_exp*exp

    # Each row represents a sample, each column is the class conditional probability for that sample
    p_of_x_given_y = np.zeros((N,k))

    # Final probability we care about
    p_of_y_given_x = np.zeros((N,k))

    ypred = np.zeros((N, 1))

    # Loop through all samples
    for i in range(0, N):
        sample = Xtest[i]
        # For each sample, calculate the probabilities of each class
        for my_class in classes:
            p_of_x_given_y[i, my_class-1] = calculate_pdf(d, covmats[my_class-1], sample, means[my_class-1])

        # Calculate p_of_y_given_x
        p_of_y_given_x[i] = (p_of_x_given_y[i] / np.sum(p_of_x_given_y[i]))

        # Get the maximum probability (add 1 since the classes aren't 0 indexed)
        ypred[i] = np.argmax(p_of_y_given_x[i]) + 1

    # Calculate accuracy
    acc = 0
    match, garbage = np.where(ypred == ytest)
    acc = np.shape(match)[0]/np.shape(ytest)[0]

    #compare = np.concatenate((ypred, ytest), axis = 1)
    #print(compare)

    # IMPLEMENT THIS METHOD
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1
    
    #based off equation on handout c1 page 6
    """rows = X.shape[0]
    columns = X.shape[1]
    w = np.ones((columns, 1))
    for hg in range (0,100):
        for j in range (0,columns):
            sumval = 0
            for i in range (0,rows):
                sumval += (np.inner(w.T, X[i,])-y[i,])*X[i,j]
            w[j,] = w[j,] - (.005)*sumval #we'll have to experiment for optimal alpha I assume.
    # IMPLEMENT THIS METHOD"""
    columns = X.shape[1]
    w = np.dot(np.dot(inv(np.dot(X.T,X)),X.T),y)
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d
    # y = N x 1
    # lambd = ridge parameter (scalar)
    # Output:
    # w = d x 1
    
    #this equation is found on page 8 in handout C1 for MAP estimate of w using Ridge Regression
    #I had to use np.dot to avoid an error
    columns = X.shape[1]
    I = np.identity(columns)
    w = np.dot(np.dot(inv(I*lambd + np.dot(X.T,X)),X.T),y)
    # IMPLEMENT THIS METHOD
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    rows = Xtest.shape[0]
    sumval = 0
    for i in range (0,rows):
        sumval +=(ytest[i,]-np.inner((w.T),Xtest[i,]))**2
    mse = sumval*(1/rows)
    
    # IMPLEMENT THIS METHOD
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda

    wm = np.asmatrix(w)
    wm = w.T
    ycol =  np.reshape(y, (np.shape(y)[0]))

    rows = X.shape[0]
    cols = X.shape[1]

    xw = np.dot(X, w)
    middle = ycol - xw
    error = 0.5 * np.dot(middle, middle)

    # Add the regularization term, (1/2)labmda * w^T*w

    regularization = 0.5 * lambd * np.dot(w, w)
    error += regularization

    # Now we calculate the gradient of the error function
    # dJ(w)/dw_j = sum from i = 1 to N[w.T*x_i - y_i) * x_ij

    xtw = np.dot(X, w)
    inner = ycol - xtw
    first = -1 * np.dot(X.T, inner)

    reg = lambd * w
    error_grad = first + reg

    print("Objective Function: " + str(error))
    
    # IMPLEMENT THIS METHOD
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xd - (N x (d+1))
    Xn = x.shape[0]
    Xd = np.zeros((Xn,p+1))
    for j in range (0,Xn):
        for i in range (0,p+1):
            val = pow(x[j],i)
            Xd[j,i] = val
    # IMPLEMENT THIS METHOD
    return Xd

# Main script

# Problem 1
# load the sample data
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')
    
# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))

# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

#print(xx.shape[0])
#exit()

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
#print(zqdares)
#print(np.unique(zqdares))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()

# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()

# Problem 5
pmax = 7
lambda_opt = .055 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    print(X[:,2])
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()