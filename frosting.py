#Import tools.
#___________________________________________________________________________________
import numpy as np 
import statistics

#Import Boston Housing Dataset.
#___________________________________________________________________________________
from sklearn import datasets
from sklearn.datasets import load_boston
boston=load_boston()
#boston.feature_names.shape #Boston has 13 features.
#boston.data.shape #506 and 13
#boston.target

#Import the Digits Dataset.
#___________________________________________________________________________________
from sklearn.datasets import load_digits
digits=load_digits()
#digits.keys()
#digits.data.shape #(1797 and 64)

# Create Boston50 Dataset.
#___________________________________________________________________________________
statistics.median(boston.target)

T50=statistics.median(boston.target)
list=[]
index=0
while index<len(boston.target):
    for index, value in enumerate(boston.target):
        if value>=T50:
            list.append(1)
        else:
            list.append(0)
    index+=1
list_array = np.asarray(list)

#This is the Boston 50 dataset. 
random_array=np.random.rand(boston.data.shape[0])
split=random_array<np.percentile(random_array,70) #Makes it such that 70% of data goes into training, 30% testing.
data_train=boston.data[split]
target_train=list_array[split]
data_test=boston.data[~split]
target_test=list_array[~split]
import numpy as np

#Create Boston75 Dataset.
#___________________________________________________________________________________
T75=np.percentile(boston.target,75) #Returns 75% percentile, which is 25.0

list2=[]
index=0
while index<len(boston.target):
    for index, value in enumerate(boston.target):
        if value>=T75:
            list2.append(1)
        else:
            list2.append(0)
    index+=1
list_array2 = np.asarray(list2)

#This is the Boston 75 dataset.
random_array=np.random.rand(boston.data.shape[0])
split=random_array<np.percentile(random_array,70)
data_train2=boston.data[split]
target_train2=list_array2[split]
data_test2=boston.data[~split]
target_test2=list_array2[~split]

#Prepare the Digits dataset.
#___________________________________________________________________________________
#This is the digits dataset.
random_array=np.random.rand(digits.data.shape[0])
split=random_array<np.percentile(random_array,70)
data_train4=digits.data[split]
target_train4=digits.target[split]
data_test4=digits.data[~split]
target_test4=digits.target[~split]

#Define class(MyLogisticReg2).
#___________________________________________________________________________________
class MyLogisticReg2:
    
    #Initializes the parameters w and w0. 
    def __init__(self, d, lr, n_iters):  
        #Store these values.
        #d is the number of dimensions/features. 
        self.d=d #For boston, d=13. 
        self.lr=lr
        self.n_iters=n_iters
        
        #Create some weights. Set them to none at first.
        #Create the bias. Set it to none. 
        #We will have to come up with them.
        self.weights=None
        self.bias=None
    
    #Develop a fit method. 
    #This is the training step and involves gradient descent.
    #x is a numpy vector of size m*n where m is the number of samples and n is the number of features for each sample.
    #y is also of size m. Each training sample has 1 vector.
    def fit(self, X, y):
        #We need to initialize the weights/our parameters. 
        #Initialize the parameters.
        n_samples, n_features = X.shape
        #Initialize the weights by creating a vector of only 0's. It's size is the number of features.
        self.weights = np.zeros(n_features)
        #Set the bias to 0 at first.
        #Note: you can also use random numbers for the initialization, but 0 is just fine.
        self.bias=0
        
        #Use gradient descent. Iteratively update the weights.
        for _ in range(self.n_iters): #n_iters is the number of iterations we want to have.
            linear_model = np.dot(X, self.weights) + self.bias #This is wx+b. Use np.dot to multiple the vectors. 
            #Then apply the sigmoid function. Apply a helper method below.
            y_predicted = self.sigmoid(linear_model) #This is our approximation of y.
            #Update our weights using the update rules.
            
            #This is the derivative with respect to w.
            dw=(1/n_samples) * np.dot(X.T, (y_predicted-y)) #y predicted minus the actual y.
            
            #The derivative with respect to bias is the same but without the x. 
            db=(1/n_samples)*np.sum(y_predicted-y)
            
            #Now that we have our derivatives, update the parameters.
            self.weights-=self.lr * dw
            self.bias-=self.lr * db
        #set_trace() 
        
    #Develop a predict method. Input the new test samples that you want to predict. 
    def predict(self, X):
        #First, approximate the data using a linear model.
        linear_model=np.dot(X, self.weights) + self.bias
        #Then apply a sigmoid function to gete th probability.
        y_predicted=self.sigmoid(linear_model)
        #Predict the y class. Use a list comprehension. 
        y_predicted_cls=[1 if i > 0.5 else 0 for i in y_predicted] #Do this for each value in y_predicted.
        return y_predicted_cls
    
    def sigmoid(self, linear_model):
        return 1/(1+np.exp(-linear_model))

#Store my logistic regression function in variables so that they can be referenced in my k fold cross validation function.
#___________________________________________________________________________________
BOSTON=MyLogisticReg2(d=13, lr=0.01, n_iters=1000)
#BOSTON.fit(data_train, target_train)
#BOSTON.predict(data_test) #length is 354
#len(data_test) #152

#Create k fold cross validation function.
#This function performs k fold cross validation on X and y using method and returns the error rate in each fold.
#The method used is my logistric regression function.
#___________________________________________________________________________________
#X will be boston.data, boston.data, digits.data
#y will be list_array, list_array2, digits.target

def my_cross_val(X, y, k):
    from sklearn.metrics import accuracy_score
    error_rate=[0 for x in range(k)] #Or error_rate=np.zeros(10)
    for i in range(k):
        random_array=np.random.rand(X.shape[0])
        split=random_array<np.percentile(random_array,70)
        data_train3=X[split]
        target_train3=y[split]
        data_test3=X[~split]
        target_test3=y[~split]
        
        BOSTON.fit(data_train3, target_train3)
        #set_trace()
        y_prediction=BOSTON.predict(data_test3)
        #set_trace()
        error_rate[i]=(1-accuracy_score(target_test3, y_prediction)) #Output is the error.
    return (error_rate, np.mean(error_rate), statistics.stdev(error_rate))
    #Will report the error rates across folds, mean across the error rates, standard deviation.

#Apply the cross validation code to the datasets.
#___________________________________________________________________________________
print('Error rates for MyLogisticReg2 on Boston50')
#from IPython.core.debugger import set_trace
my_cross_val(X=boston.data, y=list_array, k=5)

print('Error rates for MyLogisticReg2 on Boston75')
my_cross_val(X=boston.data, y=list_array2, k=5)

#Prepare sklearn's default logistic regression model.
#I am going to compare the imputations from this model with those calculated using my LR model.
#___________________________________________________________________________________
from sklearn.linear_model import LogisticRegression

myLR=LogisticRegression(penalty='l2',solver='lbfgs', multi_class='multinomial', max_iter=5000)

#Create a second k fold cross validation function.
#This function uses sklearn's default LR model.
#___________________________________________________________________________________
#X will be boston.data, boston.data, digits.data
#y will be list_array, list_array2, digits.target
def my_cross_val2(method, X, y, k):
    from sklearn.metrics import accuracy_score
    error_rate=[0 for x in range(k)] #Or error_rate=np.zeros(10)
    for i in range(k):
        random_array=np.random.rand(X.shape[0])
        split=random_array<np.percentile(random_array,70)
        data_train3=X[split]
        target_train3=y[split]
        data_test3=X[~split]
        target_test3=y[~split]
        
        method.fit(data_train3, target_train3)
        #set_trace()
        y_prediction=method.predict(data_test3)
        #set_trace()
        error_rate[i]=(1-accuracy_score(target_test3, y_prediction)) #Output is the error.
    return (error_rate, np.mean(error_rate), statistics.stdev(error_rate))
    #Will report the error rates across folds, mean across the error rates, standard deviation.

#Apply the second cross validation function to the sklearn model.	
#___________________________________________________________________________________
print('Error rates for LogisticRegression on Boston50')
my_cross_val2(method=myLR, X=boston.data, y=list_array, k=5)

print('Error rates for LogisticRegression on Boston75')
my_cross_val2(method=myLR, X=boston.data, y=list_array2, k=5)
