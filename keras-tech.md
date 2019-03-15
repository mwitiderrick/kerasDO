# How to Build a Deep Learning Model to Predict Employee Retention Using Keras and TensorFlow

### Introduction
[Keras](https://keras.io/) is a neural network API that is written in Python. It runs on top of [TensorFlow](https://www.tensorflow.org/), [CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/), or [Theano](http://deeplearning.net/software/theano/). It is a high-level abstraction of these deep learning frameworks and therefore makes experimentation faster and easier. Keras is mainly used for solving problems involving large datasets. Keras is modular and easier to implement compared to other deep learning frameworks e.g PyTorch. TensorFlow is a great choice for the model you are going to build for a couple of reasons. One is because it works efficiently with computation involving arrays since you will be converting the dataset you'll use for this exercise into arrays. Secondly, it allows for the execution of code on either CPU or GPU. This enables you to run your model on a GPU especially in the event that you are working with a massive dataset. Thirdly TennsorFlow has a huge community that can support you in the event that you encounter challenges while building your model.


Retaining the best employees is an important factor for most organizations. Using this dataset available at [Kaggle](https://www.kaggle.com/liujiaqi/hr-comma-sepcsv#HR_comma_sep.csv) you'll build a deep learning model that will predict the probability that of an employee will leave a company.  The dataset has features that measure employee satisfaction in a company. We'll use these feature in building a model that determines the probability of an employee leaving the company. To build this model, you'll use the Keras Sequential layer to build the different layers for the model.


## Prerequisites

Before we begin this guide you'll need the following:

- A [local Python 3 development environment](https://www.digitalocean.com/community/tutorial_series/how-to-install-and-set-up-a-local-programming-environment-for-python-3), including [pip](https://pypi.org/project/pip/), a tool for installing Python packages, and [venv](https://docs.python.org/3/library/venv.html), for creating virtual environments.

- Have set up Jupyter Notebook to use in this tutorial [Jupyter Notebook](https://www.digitalocean.com/community/tutorials/how-to-set-up-jupyter-notebook-for-python-3).

- Familiarity with [Machine learning](https://www.digitalocean.com/community/tutorials/an-introduction-to-machine-learning).

## Step 1 — Data Pre-processing

In this step, you'll load in the `pandas` module for manipulating your data and NumPy so as to later convert the data into NumPy arrays. After that, you will convert all the columns that are in string format to numerical values. They have to be converted into numerical values because computers only work with numbers.

Before you begin this step, move to the environment you've created for this project as part of the prerequisites. Install Keras and TensorFlow via a pip command. Add this code to a command line or terminal and run it.

```custom_prefix((my_env))
pip install keras tensorflow
```

_Data Pre-processing_ is necessary to prepare your data in a manner that deep learning model can accept. If there are _categorical variables_ you have to convert them to numbers because the algorithm only accepts numerical figures. A categorical variable represents quantitive data represented by names. You'll start by loading in the dataset using `pandas` — the data manipulation package in Python.

Now, activate your environment and open Jupyter notebook so that you can get started. You'll import the [modules](https://www.digitalocean.com/community/tutorials/how-to-import-modules-in-python-3) you'll need for the project and then load the dataset in a notebook cell. Add the following code to a notebook cell and then run it.

![Importing Modules](https://i.imgur.com/R7ueVRJ.png)


```
import pandas as pd
import numpy as np
df = pd.read_csv("https://raw.githubusercontent.com/mwitiderrick/kerasDO/master/HR_comma_sep.csv")
```
At this point you have loaded NumPy and Pandas, then used Pandas to load in the dataset that you will use.

You can get a glimpse at the dataset we're working with by checking its ` head()`. Add the following code to a notebook cell and then run it.

```
df.head()
```
![Alt Checking the head for the dataset](https://i.imgur.com/mgkoAJy.png)

You'll now proceed to convert the categorical columns to numbers. You do this by converting them to dummy variables. Dummy variables are usually ones and zeros that indicate the presence or absence of a categorical feature. In this kind of situations, you also avoid the dummy variable trap by dropping the first dummy.

<$>[note]
**Note:** The dummy variable trap is a situation whereby two or more variables are highly correlated. This leads to our model performing poorly. We, therefore, drop one dummy variable to always remains with N-1 dummy variables. Any of the dummy variables can be dropped because there is no preference as long as we remain with N-1 dummy variables. An ideal example is if we have a male or female column. When we create the dummy variable we shall get two columns; a male column and a female column. We can drop one of the columns because if one isn't male then they are female.
<$>

As you have done previously, you will input the next code in a Jupyter Notebook cell and run it.

![Creating Dummy Variables](https://i.imgur.com/rHbBkaq.png)

```
feats = ['department','salary']
df_final = pd.get_dummies(df,columns=feats,drop_first=True)

```

At this point, you have loaded in the dataset and converted the salary departments into a format that can be accepted by the Keras deep learning model. In the next step, you will split the dataset into a training and testing set.

## Step 2 — Training and Testing Split Separation

In this step, you'll use [Scikit-learn](https://scikit-learn.org/) to split the dataset into a training and testing set. You'll use the training set to train your deep learning model and the test set to test it. This is necessary since you will use some part of the employee data to train the model and a part of it to test its performance. You'll start by importing the `train_test_split` module from the Scikit-learn package. This is the module that will provide you with the splitting functionality.

You'll kick it off by putting this code in the next Jupyter Notebook cell and running it.

![Training and Testing Split Separation](https://i.imgur.com/LJ7g9PQ.png)
```
from sklearn.model_selection import train_test_split
```

Now that you have imported `train_test_split` module, you are going to use the `left` column in your dataset to predict if an employee will leave the company. You, therefore, have to ensure that your deep learning model doesn't come into contact with this column. This will prevent your model from memorizing the results from the dataset and giving you wrong predictions. You, therefore, drop the `left` column.

Your deep learning model expects to get the data as arrays. You, therefore, use [NumPy] (http://www.numpy.org/) to convert the data to NumPy arrays. This is achieved by adding the `.values` attribute at the end.

Now add this code to the Jupyter Notebook cell and run it. It will drop the left column and since you have used the .values attribute, it will return the values as NumPyu arrays.

![Dropping the final column](https://i.imgur.com/I3nwzlN.png)

```
X = df_final.drop(['left'],axis=1).values
y = df_final['left'].values

```

At this point, you are now ready to convert the dataset into testing and training set. You'll use 70% of the data for training and 30% for testing. The training ratio is more than the testing ration because you have to make sure that most of the data is used for the training process. You can also experiment with a ration of 80% for the training set and 20% for the testing set. Now add this code to the Jupyter Notebook cell and run it.

![Setting the ratio for train and test set](https://i.imgur.com/pEeA0Is.png)
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```
At this point, you already have the data in the manner that Keras expects it to be in, i.e NumPy arrays. Your data also already split into a training and testing set. You'll pass this data to the Keras model in a moment. However, before you can do that you have to transform the data as you will see in the next step.


## Step 3 — Transforming the Data
In this step you will scale the data using the `StandardScaler`. When building deep learning models it is usually good practice to scale your dataset in order to make the computations that will take place efficient. You'll use Scikit-learn's `StandardScaler` to scale the features to be within the same range. This will transform the values to have a mean of 0 and a standard deviation of 1. This step is important because you are comparing features that have different measurements. This is step is usually a requirement in machine learning. Now add this code to the Jupyter Notebook cell and run it.

![Data Transformation](https://i.imgur.com/UD37IFI.png)

```
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```
At this point all your dataset features have been scaled in order to be within the same range. With this in place, you can start building the artificial neural network in the next step.

## Step 4 — Building the Artificial Neural Network
In this step you will use Keras to build the deep learning model. In order to do this, you need to import Keras. By default, Keras will use the TensorFlow backend.  From Keras, you'll then import the `Sequential` and `Dense` modules. `Sequential` will be used to initialize the artificial neural network while `Dense` is for adding layers to your deep learning model. An artificial neural network is a computational model that is built using inspiration from the working of the human brain. When building a deep learning model you usually specify three types of layers. The first is the input layer that contains the features and the last one is the output layer. The layers in between are known as the hidden layers. Now add this code to the Jupyter Notebook cell and run it.

![Building the Artificial Neural Network](https://i.imgur.com/JvGz0QC.png)

```
import keras
from keras.models import Sequential
from keras.layers import Dense
```

You'll now use `Sequential` to initialize linear stack of layers. Later on, you can add more layers using `Dense`. Since this is a classification problem, you'll create a classifier variable. Now add this code to the Jupyter Notebook cell and run it.

![Initializing the layers](https://i.imgur.com/XHhQ8VX.png)

```
classifier = Sequential()
```

After this, you can now start adding layers to your network. You do this by using the `.add()` function on your classifier and specifying a couple of parameters:
- The first parameter is the number of nodes that your network should have. The connection between different nodes is what forms the neural network. One of the strategies to determine the number of nodes is to take the average of the nodes in the input layer and the output layer.

- The second parameter is the `kernel_initializer.` When you fit your deep learning model the weights will be initialized to numbers close to zero but not zero. To achieve this you use the uniform distribution initializer.  `kernel_initializer` is the function that initializes the weights.

- The third parameter is the activation function. Your deep learning model will learn through this function.  There are usually linear and non-linear activation functions. You use the ReLU activation function because it generalizes well on your data. Linear functions are not good for problems like these because they form a straight line.

- The last parameter is `input_dim` which represents the number of features in our dataset.

Now add this code to the Jupyter Notebook cell and run it.

![Adding the first input layer](https://i.imgur.com/k7sIK8H.png)

```
classifier.add(Dense(9, kernel_initializer = "uniform",activation = "relu", input_dim=18))
```

The next thing you do is add the output layer that will give us the predictions. The output layer takes the following parameters:
- The number of output nodes. You expect to get one output i.e if an employee leaves the company. Therefore you specify one output node.

- For `kernel_initializer` we use the Sigmoid activation function so that we can get the probability that an employee will leave. In the event that we were dealing with more than two categories, we would use the Softmax activation function which is a variant of the Sigmoid activation function.

Now add this code to the Jupyter Notebook cell and run it.

![Adding the first input layer](https://i.imgur.com/TUA2c9O.png)

```
classifier.add(Dense(1, kernel_initializer = "uniform",activation = "sigmoid"))

```

Next, you apply a gradient descent to the neural network. This an optimization strategy that works to reduce errors during the training process. There are several types of optimization strategies but you'll use a popular one known as `adam` in this tutorial. Applying gradient descent is done via the `compile` command which also takes a couple of other parameters as explained below:

- `optimizer` is the gradient descent
- The second parameter is the `loss` function that will be used in the gradient descent. Since this is a binary classification problem we use the `binary_crossentropy` loss function.

- The last parameter is the metric that we'll use to evaluate your model. In this case, you'd like to evaluate it based on its accuracy when making predictions.

Now add this code to the Jupyter Notebook cell and run it.

![Compiling the model](https://i.imgur.com/ZiwyMuA.png)


```
classifier.compile(optimizer= "adam",loss = "binary_crossentropy",metrics = ["accuracy"])

```

At this stage, you are now ready to fit your classifier to your dataset. Keras makes this possible via the `.fit()` method. It takes a couple of parameters as explained below:

- The first parameter is the training set with the features

- The second parameter is the column that we are making the predictions on

- The `batch_size` represents the number of samples that will go through the neural network at each training round.

- `epochs` represents the number of times that the dataset will be passed via the neural network. The more the epochs the longer it will take to run our model which also gives us better results.

Now add this code to the Jupyter Notebook cell and run it so as to fit the model to your datset.

![Fitting the dataset](https://i.imgur.com/5Xwezaw.png)

```
classifier.fit(X_train, y_train, batch_size = 10, epochs = 1)
```
In this step you have created the deep learning model, compiled it and fitted it to your dataset. You are now ready to make some predictions using the deep learning model. In the next step you will start making predictions with dataset that the model hasn't yet seen.

## Step 5— Running Predictions on the Test Set
In this step you will use the testing dataset to make predictions using the model that you worked on in the previous step. Keras enables you to make predictions by using the `.predict()` function. Now add this code to the Jupyter Notebook cell and run it.

![Making Predictions](https://i.imgur.com/jPLuOGo.png)

```
y_pred = classifier.predict(X_test)
```
This will give you the probabilities of an employee leaving. You'll work with a probability of 50% and above to indicate a high chance of the employee leaving the company. Enter the following line of code in your Jupyter Notebook cell in order to set this threshold.
Now add this code to the Jupyter Notebook cell and run it in order to set that threshold.

![setting prediction threshold](https://i.imgur.com/BqxKFvc.png)

```
y_pred = (y_pred > 0.5)
```
In this step, you have created predictions using the predict method. You have also set the threshold for determining if an employee is likely to leave. In the next step, you will use a confusion matrix to evaluate how well the model performed on the predictions.

## Step 6 — Checking the Confusion Matrix
In this step, you will use a confusion matrix to check the number of correct and wrong predictions.  You will use these figures to calculate the model accuracy. To achieve this you use a confusion matrix that is provided by Scikit-learn. Now add this code to the Jupyter Notebook cell and run it.

```
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
```
![Checking the Confusion Matrix](https://i.imgur.com/CbWWQ7U.png)

The confusion matrix below means that your deep learning model made 3305 + 375 correct predictions and 106 + 714 wrong predictions. You can calculate the accuracy has (3305 + 375) / 4500. 4500 is the total number of observations in our dataset. This gives us an accuracy of 81.7%.

```
[secondary_label Output]
array([[3305,  106],
       [ 714,  375]])
```
In this step, you have seen how to evaluate your model using the confusion matrix. In the next step, you will work on making a single prediction using the model that you have developed.
## Step 7 — Making a Single Prediction
In this step, you will use your model to make a single prediction given the details of one employee. You will achieve this by predicting the probability of a single employee leaving the company. All you have to do is pass this employee's features to the predict method. Just like you did earlier you have to scale the features as well and convert them to a NumPy array. Now add this code to the Jupyter Notebook cell and run it.

![Making a Single Prediction](https://i.imgur.com/6QBWjvT.png)
```
new_pred = classifier.predict(sc.transform(np.array([[0.26,0.7 ,3., 238., 6., 0.,0.,0.,0., 0.,0.,0.,0.,0.,1.,0., 0.,1.]])))
```
You then set a threshold of 50% and check the predicted value. You can see in this case that the employee won't leave the company. Now add this code to the Jupyter Notebook cell and run it.

![Setting the prediction threshold](https://i.imgur.com/72OlVVZ.png)

```
new_pred = (new_pred > 0.5)
new_pred
```

```
[secondary_label Output]
array([[False]])
```
In this step, you have seen how to make a single prediction given the features of a single employee. In the next step, you will work on improving the accuracy of your model.

## Step 8 — Improving  the Model Accuracy
In this step, you will use K-fold cross-validation to improve the accuracy of the model that you built earlier. You notice that if you train the model many times we keep getting different results. The accuracies for each training have a high variance. In order to solve this problem, you use the K-fold cross-validation. Usually, K is set to 10. In this technique, the model is trained on the first 9 folds and tested on the last fold. This iteration continues until all folds have been used. Each of the iterations gives its own accuracy. The accuracy of the model becomes the average of all these accuracies.

Keras enables you to implement K-fold cross-validation via the `KerasClassifier` wrapper. This wrapper is from Scikit-learn's cross-validation. You'll start by importing the `cross_val_score` cross-validation function and the `KerasClassifier`. Now add this code to the Jupyter Notebook cell and run it.

![Improving  the Model Accuracy](https://i.imgur.com/xUhxj5F.png)

```
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
```
The first step is to create a function that will be passed to the `KerasClassifier`. This is one of the arguments it expects. The function is a wrapper of the neural network design that we used earlier. The parameters passed are similar to the ones used earlier.
Now add this code to the Jupyter Notebook cell and run it in order to create that function.

![Creating the classifier function](https://i.imgur.com/vcQiQlP.png)

```
def make_classifier():
    classifier = Sequential()
    classiifier.add(Dense(9, kernel_initializer = "uniform", activation = "relu", input_dim=18))
    classifier.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))
    classifier.compile(optimizer= "adam",loss = "binary_crossentropy",metrics = ["accuracy"])
    return classifier
```
The `KerasClassifier` takes three arguments:
- `build_fn` the function with the neural network design
- `batch_size` the number of samples to be passed via the network in each iteration
- `nb_epoch` the number of epochs the network will run

Now add this code to the Jupyter Notebook cell and run it.

![setting the KerasClassifier function](https://i.imgur.com/fIktN6V.png)

```
classifier = KerasClassifier(build_fn = make_classifier, batch_size=10, nb_epoch=1)
```
Next, you apply the cross-validation using Scikit-learn's `cross_val_score`. This function will give you 10 accuracies since you have specified the number of folds as 10. You, therefore, assign it to the accuracies variable and later use it to compute the mean accuracy. It takes the following arguments:
- `estimator` which is the classifier that we defined above
- `X` the training set features
- `y` the value to be predicted in the training set
- `cv` the number of folds
- `n_jobs` the number of CPUs to use. Specifying it as -1 will make use of all the available CPUs

Now add this code to the Jupyter Notebook cell and run it.

![Applying cross-validation using Scikit-learn](https://i.imgur.com/AicexkN.png)

```
accuracies = cross_val_score(estimator = classifier,X = X_train,y = y_train,cv = 10,n_jobs = -1)
```
Once this has run, you can now compute the mean and variance of the accuracies.
Now add this code to the Jupyter Notebook cell and run it.

![Computing the mean accuracies](https://i.imgur.com/HKoBP6o.png)

```
mean = accuracies.mean()
mean
```
```
[secondary_label Output]
0.8343617910685696
```
Now add this code to the Jupyter Notebook cell and run it.

![Computing the mean variance](https://i.imgur.com/vCx0w1B.png)

```
variance = accuracies.var()
variance
```
```
[secondary_label Output]
0.0010935021002275425
```
In this step, you have improved your model's accuracy by using K-Fold Cross-Validation. In the next step, you will work on the overfitting problem.

## Step 9 — Adding Drop out Regularization to Fight Over-Fitting
In this step, you will add a layer to your model that will help you fight over-fitting in your model. Predictive models are prone to a problem known as overfitting. This is a scenario whereby the model memorizes the results in the training set and isn't able to generalize on data that it hasn't seen.  Overfitting can be observed from having a very high variance on the accuracies. In neural networks, the technique used to fight to overfit is known as dropout regularization. This is achieved by adding a `Dropout` layer in our neural network. It has a `rate` parameter which indicates the number of neurons that will be deactivated at each iteration. The process of deactivating them is usually random. In this case, we specify 0.1 as the rate meaning that 1% of the neurons will be deactivated during the training process. The network design remains the same.
Now add this code to the Jupyter Notebook cell and run it.

![Adding Drop out Regularization to Fight Over-Fitting](https://i.imgur.com/NWVOZ52.png)

```
from keras.layers import Dropout

classifier = Sequential()
classifier.add(Dense(9, kernel_initializer = "uniform", activation = "relu", input_dim=18))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))
classifier.compile(optimizer= "adam",loss = "binary_crossentropy",metrics = ["accuracy"])

```
In this step, you have added a Dropout layer in the model in order to fight overfitting. In the next step  you will work on further improving the model by tuning the parameters you used while creating the model.

## Step 10 — Hyperparameter Tuning
In this step, you will use Grid Search to search for the best parameters for your deep learning model. This will help in improving model accuracy. Grid Search is the technique that is used to experiment with different model parameters in order to obtain the ones that give us the best accuracy. Scikit-learn provides the `GridSearchCV` function to enable this functionality.  You will now proceed to modify the `make_classifier`  function in order to try out different parameters. Now add this code to the Jupyter Notebook cell and run it.

![Hyperparameter Tuning](https://i.imgur.com/z8a3Rij.png)

```
from sklearn.model_selection import GridSearchCV
def make_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(9, kernel_initializer = "uniform", activation = "relu", input_dim=18))
    classifier.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))
    classifier.compile(optimizer= optimizer,loss = "binary_crossentropy",metrics = ["accuracy"])
    return classifier
```
The next step is to define the classifier like you did before. Enter this code into a Jupyter Notebook cell and run it.

![Defining the classifier](https://i.imgur.com/yCUYKrb.png)

```
classifier = KerasClassifier(build_fn = make_classifier)
```
You will now proceed to set a couple of parameters that you wish to experiment with. You'll try out different batch sizes, number of epochs and different types of optimizer functions. Enter this code into a Jupyter Notebook cell and run it.

![Defining Parameters](https://i.imgur.com/xn6LqYG.png)
```
params = {
    'batch_size':[20,35],
    'nb_epoch':[5,10],
    'Optimizer':['adam','rmsprop']
}
```
The grid search function expects the following parameters:

- `estimator` which is the classifier that we are using
- `param_grid` the set of parameters that we are going to test.
- `scoring` the metric we are going to use
- `cv` the number of folds to test on

Enter this code into a Jupyter Notebook cell and run it.

![Searching for best parammeters](https://i.imgur.com/xn6LqYG.png)

```
grid_search = GridSearchCV(estimator=classifier,  param_grid=params,
                        scoring="accuracy", cv=10)
```

Next, you fit this grid search to your training dataset. Enter this code into a Jupyter Notebook cell and run it.

![Fititing the grid search](https://i.imgur.com/vaX7P2L.png)

```
grid_search = grid_search.fit(X_train,y_train)

```
```
[secondary_label Output]
Epoch 1/1
9449/9449 [==============================] - 2s 205us/step - loss: 0.5150 - acc: 0.7848
Epoch 1/1
9449/9449 [==============================] - 2s 210us/step - loss: 0.4971 - acc: 0.7808
Epoch 1/1
9449/9449 [==============================] - 2s 214us/step - loss: 0.5051 - acc: 0.7807
Epoch 1/1
9449/9449 [==============================] - 2s 218us/step - loss: 0.5037 - acc: 0.7775
```

Now you can obtain the best parameters from this search using the `best_params_` attribute. Enter this code into a Jupyter Notebook cell and run it.

![Obtaining the best parammeters](https://i.imgur.com/gP7dXsv.png)

```
best_param = grid_search.best_params_
best_accuracy = grid_search.best_score_
```
You can now check the best parameters for our model. You notice that the best batch size is 20, the best number of epochs is 10 and the adam optimizer is the best for our model. Enter this code into a Jupyter Notebook cell and run it.

```
best_param
```
```
[secondary_label Output]
{'batch_size': 20, 'nb_epoch': 10, 'optimizer': 'adam'}
```
You can now check the best accuracy for our model. Enter this code into a Jupyter Notebook cell and run it.

```
best_accuracy
```
```
[secondary_label Output]
0.8235070006667302
```


## Conclusion
In this tutorial, we have learned how we can use [Keras](https://keras.io) to build an artificial neural network that predicts the probability that an employee will leave a company. We combined our previous knowledge in machine learning using `Scikit-Learn` to achieve this. There are a couple of things you can do to keep improving this model:
- Try different activation functions
- Use different optimizer functions
- Experiment with a different number of folds

For more practice, you can try a different dataset and see the results you obtain.
