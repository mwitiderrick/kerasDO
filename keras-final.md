<!--NOTE Hi Derrick,

When going through my comments and implementing revisions, please leave my comments in place — this makes the process a lot quicker for me to review the next draft.

At the moment your explanations are really high level, and while we want those and want to expand them, the reader also needs some detail on the lower code level. You're adding the python incrementally, so take the time to explain parts of the code to them to demonstrate the higher level points you're making.

* Please double-check spelling in code blocks.
* Value Error in Step 10 (noted), I could not complete the tech test. Please review this area. And detail the fix for me.
* Deepen your explanations throughout, I've marked where this is necessary. Feel free to _also_ provide links to further reference material on any of the concepts you explain.

-->


# How to Build a Deep Learning Model to Predict Employee Retention Using Keras and TensorFlow

### Introduction
[Keras](https://keras.io/) is a neural network API that is written in Python. It runs on top of [TensorFlow](https://www.tensorflow.org/), [CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/), or [Theano](http://deeplearning.net/software/theano/). It is a high-level abstraction of these deep learning frameworks and therefore makes experimentation faster and easier. Keras is mainly used for solving problems involving large datasets. Keras is modular and easier to implement compared to other deep learning frameworks.

<!--TODO As you have for Keras, can you explain what Tensorflow is, as the first sentence in this paragraph -->
TensorFlow is an end-to-end open source platform for machine learning. TensorFlow is a great choice for the model you are going to build for a couple of reasons. One is because it works efficiently with computation involving arrays since you will be converting the dataset you'll use for this exercise into arrays. Secondly, it allows for the execution of code on either CPU or GPU. This enables you to run your model on a GPU especially in the event that you are working with a massive dataset. Thirdly TensorFlow has a huge community that can support you in the event that you encounter challenges while building your model.


In this tutorial, you'll build a deep learning model that will predict the probability that of an employee will leave a company. Retaining the best employees is an important factor for most organizations. To build your model, you'll use this dataset available at [Kaggle](https://www.kaggle.com/liujiaqi/hr-comma-sepcsv#HR_comma_sep.csv), which has features that measure employee satisfaction in a company. You'll use these feature in building a model that determines the probability of an employee leaving the company. To build this model, you'll use the Keras Sequential layer to build the different layers for the model.

## Prerequisites

Before we begin this guide you'll need the following:

- A [local Python 3 development environment](https://www.digitalocean.com/community/tutorial_series/how-to-install-and-set-up-a-local-programming-environment-for-python-3), including [pip](https://pypi.org/project/pip/), a tool for installing Python packages, and [venv](https://docs.python.org/3/library/venv.html), for creating virtual environments.

- Have set up Jupyter Notebook to use in this tutorial [Jupyter Notebook](https://www.digitalocean.com/community/tutorials/how-to-set-up-jupyter-notebook-for-python-3).

- Familiarity with [Machine learning](https://www.digitalocean.com/community/tutorials/an-introduction-to-machine-learning).

## Step 1 — Data Pre-processing

In this step, you'll load in the `pandas` module for manipulating your data and NumPy for converting the data into NumPy arrays. After that, you will convert all the columns that are in string format to numerical values for your computer to process.

Before you begin this step, move to the environment you've created for this project as part of the prerequisites. Install Keras and TensorFlow via a `pip` command:

```custom_prefix((my_env))
pip install keras tensorflow
```

_Data Pre-processing_ is necessary to prepare your data in a manner that a deep learning model can accept. If there are _categorical variables_ you have to convert them to numbers because the algorithm only accepts numerical figures. A _categorical variable_ represents quantitive data represented by names. You'll start by loading in the dataset using `pandas` — the data manipulation package in Python.

Now, activate your environment and open Jupyter Notebook to get started. You'll import the [modules](https://www.digitalocean.com/community/tutorials/how-to-import-modules-in-python-3) you'll need for the project and then load the dataset in a notebook cell. Add the following code to a notebook cell and then run it.

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

As you have done previously, you will input the next code in a Jupyter Notebook cell and run it. <!--TODO: What is this code for? Let's tell the reader what this code is for. Please break down parts of the code after the insert to explain the construction to the reader. Otherwise, they're just copying and pasting without learning what is happening. -->

![Creating Dummy Variables](https://i.imgur.com/rHbBkaq.png)

```
feats = ['department','salary']
df_final = pd.get_dummies(df,columns=feats,drop_first=True)

```
`feats = ['department','salary']` define the two columns that we want to create dummy variables for i.e the `department` and the `salary` columns. This will generate numerical variables as required by our employee retention model.

<!--TODO: You've talked about "creating a dummy variable", but for instance what is the `feats = ['department', 'salary']` for? Make sure you're relating the technical aspects back to the employee retention model that the reader is creating. -->

![Step 1](https://i.imgur.com/W0IMyMj.png)
At this point, you have loaded in the dataset and converted the salary departments into a format that can be accepted by the Keras deep learning model. In the next step, you will split the dataset into a training and testing set.

## Step 2 — Training and Testing Split Separation

In this step, you'll use [Scikit-learn](https://scikit-learn.org/) to split the dataset into a training and testing set. You'll use the training set to train your deep learning model and the test set to test it. This is necessary since you will use some part of the employee data to train the model and a part of it to test its performance.

You'll start by importing the `train_test_split` module from the Scikit-learn package. This is the module that will provide you with the splitting functionality. Insert this code in the next Jupyter Notebook cell and run it:

![Training and Testing Split Separation](https://i.imgur.com/LJ7g9PQ.png)
```
from sklearn.model_selection import train_test_split
```

Now that you have imported `train_test_split` module, you are going to use the `left` column in your dataset to predict if an employee will leave the company. You, therefore, have to ensure that your deep learning model doesn't come into contact with this column. This will prevent your model from memorizing the results from the dataset and giving you wrong predictions. You, therefore, drop the `left` column.

Your deep learning model expects to get the data as arrays. You, therefore, use [NumPy] (http://www.numpy.org/) to convert the data to NumPy arrays. This is achieved by adding the `.values` attribute at the end.

Now add this code to the Jupyter Notebook cell and run it. It will drop the left column and since you have used the .values attribute, it will return the values as NumPy arrays. <!--NOTE: Yes! Introducing this code insert with this sentence is great! You're telling the reader what to do and what the code will do for them. -->

![Dropping the final column](https://i.imgur.com/I3nwzlN.png)

```
X = df_final.drop(['left'],axis=1).values
y = df_final['left'].values

```

At this point, you are now ready to convert the dataset into testing and training set. You'll use 70% of the data for training and 30% for testing. The training ratio is more than the testing ration because you have to make sure that most of the data is used for the training process. You can also experiment with a ration of 80% for the training set and 20% for the testing set.

Now add this code to the Jupyter Notebook cell and run it so as to split your training and testing data. <!--TODO: So this code splits the training and testing data? Let's just say this (or whatever is happening) to the reader, as they add the code. Something like "Now add this code to the cell to split your training and testing data:" -->

![Setting the ratio for train and test set](https://i.imgur.com/pEeA0Is.png)
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```
![Step 2](https://i.imgur.com/57KdWlz.png)

At this point, you already have the data in the manner that Keras expects it to be in, i.e NumPy arrays. Your data also already split into a training and testing set. You'll pass this data to the Keras model in a moment. However, before you can do that you have to transform the data as you will see in the next step.


## Step 3 — Transforming the Data
In this step, you will scale the data using the `StandardScaler`. When building deep learning models it is usually good practice to _scale_ your dataset in order to make the computations more efficient. The `StandardScaler` will ensure that your dataset values have a mean of zero and unit variable. This transforms the dataset to be normally distributed. <!--TODO: You can build out your explanation of _scaling_ here. This step is a short one, and so going into a little more depth on a method that is a vital task would add a lot to your tutorial. You can be more specific: standardizing the data is the actual action. Also, could scaling/not scaling have any effect on gradient descent? -->. You'll use Scikit-learn's `StandardScaler` to scale the features to be within the same range. This will transform the values to have a mean of 0 and a standard deviation of 1. This step is important because you are comparing features that have different measurements. This is step is usually a requirement in machine learning.


Now add this code to the Jupyter Notebook cell and run it. This scales the training set and test set. <!--TODO: Add this code to achieve what? -->

![Data Transformation](https://i.imgur.com/UD37IFI.png)

```
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

<!--TODO: Now breakdown the code here, to really delve into the python code and tell the reader how they're building the model. -->
In the code above, you started by importing the `StandardScaler` and creating an instance of it. You then used its `fit_transform` method to scale the training and testing set.

At this point, all your dataset features have been scaled in order to be within the same range. With this in place, you can start building the artificial neural network in the next step. <!--NOTE: These two sentences finish off the step well, good job. -->
 <!-- Thanks ;) -->

## Step 4 — Building the Artificial Neural Network
<!--TODO: It would be awesome if you can break these layers down into bullet points with a brief description. Then you'll really be walking through the higher level of the deep learning model. Doing this here will build a foundation for the later insertions of the different layers.-->

In this step, you will use Keras to build the deep learning model. In order to do this, you need to import Keras. By default, Keras will use the TensorFlow backend. From Keras, you'll then import the following modules

- `Sequential` that will be used to initialize the _artificial neural network_. An _artificial neural network_ is a computational model that is built using inspiration from the working of the human brain.

-  `Dense` for adding layers to your deep learning model. 

When building a deep learning model you usually specify three types of layers. The first is the input layer that contains the features and the last one is the output layer. The layers in between are known as the hidden layers.

To import `Keras`, `Sequential`, and `Dense` modules, run the following code in your notebook cell:

![Building the Artificial Neural Network](https://i.imgur.com/JvGz0QC.png)

```
import keras
from keras.models import Sequential
from keras.layers import Dense
```

You'll now use `Sequential` to initialize linear stack of layers. Later on, you can add more layers using `Dense`. Since this is a _classification problem_<!--TODO: What is a classification problem? Letting your reader understand all these elements will help the make the most of your tutorial-->, you'll create a classifier variable. A _classification problem_ is a task where you have labeled data and would like to make some predictions based on the labeled data. Now add this code to the Jupyter Notebook cell and run it.

![Initializing the layers](https://i.imgur.com/XHhQ8VX.png)

```
classifier = Sequential()
```
You have used  `Sequential` above to initialize the classifier. 

You can now start adding layers to your network. Add the following code to your notebook cell and run:

![Adding the first input layer](https://i.imgur.com/k7sIK8H.png)

```
classifier.add(Dense(9, kernel_initializer = "uniform",activation = "relu", input_dim=18))
```

You add layers using the `.add()` function on your classifier and specifying a couple of parameters:

- The first parameter is the number of nodes that your network should have. The connection between different nodes is what forms the neural network. One of the strategies to determine the number of nodes is to take the average of the nodes in the input layer and the output layer.

- The second parameter is the `kernel_initializer.` When you fit your deep learning model the weights will be initialized to numbers close to zero but not zero. To achieve this you use the uniform distribution initializer.  `kernel_initializer` is the function that initializes the weights.

- The third parameter is the activation function. Your deep learning model will learn through this function.  There are usually linear and non-linear activation functions. You use the ReLU activation function because it generalizes well on your data. Linear functions are not good for problems like these because they form a straight line.

- The last parameter is `input_dim` which represents the number of features in our dataset.


Now you'll add the output layer that will give you the predictions. Add this code to your notebook cell and run:

![Adding the first input layer](https://i.imgur.com/TUA2c9O.png)

```
classifier.add(Dense(1, kernel_initializer = "uniform",activation = "sigmoid"))
```
The output layer takes the following parameters:

- The number of output nodes. You expect to get one output i.e if an employee leaves the company. Therefore you specify one output node.

- For `kernel_initializer` you use the Sigmoid activation function so that you can get the probability that an employee will leave. In the event that we were dealing with more than two categories, you would use the Softmax activation function which is a variant of the Sigmoid activation function.


Next, you'll apply a _gradient descent_ to the neural network. This is an _optimization_ strategy that works to reduce errors during the training process. Gradient descent is how randomly assigned weights in a neural network are adjusted by reducing the cost function. A cost function is a measure of how well a neural network performs based on the output expected from it. The aim of gradient descent is to get the point where the error is the least. This is done by finding where the cost function is at the minimum usually referred to as a local minimum. Basically, in gradient descent, you differentiate to find the slope at a specific point and find out if the slope is negative or positive. This method is called gradient descent because you are descending into the minimum of the cost function. <!--TODO Can you work to develop the explanation of gradient descent in training a model? --> There are several types of optimization strategies but you'll use a popular one known as `adam` in this tutorial. Add this code to your notebook cell and run it:

![Compiling the model](https://i.imgur.com/ZiwyMuA.png)

```
classifier.compile(optimizer= "adam",loss = "binary_crossentropy",metrics = ["accuracy"])
```

Applying gradient descent is done via the `compile` command that also takes a couple of other parameters:

- `optimizer` is the gradient descent.
- `loss` is a function that you'll use in the gradient descent. Since this is a binary classification problem you use the `binary_crossentropy` `loss` function.
- The last parameter is the metric that you'll use to evaluate your model. In this case, you'd like to evaluate it based on its accuracy when making predictions.


At this stage, you're ready to fit your classifier to your dataset. Keras makes this possible via the `.fit()` method. To do this, insert the following code into your notebook and run it in order to fit the model to your dataset:

![Fitting the dataset](https://i.imgur.com/5Xwezaw.png)

```
classifier.fit(X_train, y_train, batch_size = 10, epochs = 1)
```

The `.fit()` method takes a couple of parameters:

- The first parameter is the training set with the features

- The second parameter is the column that we are making the predictions on

- The `batch_size` represents the number of samples that will go through the neural network at each training round.

- `epochs` represents the number of times that the dataset will be passed via the neural network. The more the epochs the longer it will take to run our model which also gives us better results.

![Step 4](https://i.imgur.com/wphV4sy.png)

In this step you've created your deep learning model, compiled it and fitted it to your dataset. You're ready to make some predictions using the deep learning model. In the next step, you'll start making predictions with dataset that the model hasn't yet seen.

## Step 5 — Running Predictions on the Test Set

In this step, you'll use the testing dataset to make predictions using the model that you worked on in the previous step. Keras enables you to make predictions by using the `.predict()` function.

Now add the following code to your notebook to make predictions:

![Making Predictions](https://i.imgur.com/jPLuOGo.png)

```
y_pred = classifier.predict(X_test)
```

<!--TODO Let's explain how this code is built to make predictions. -->
Since the classifier has already been trained with the training set, the above code will use the learning from the training process to make predictions on the test set.

This will give you the probabilities of an employee leaving. You'll work with a probability of 50% and above to indicate a high chance of the employee leaving the company.

Enter the following line of code in your notebook cell in order to set this threshold:

![setting prediction threshold](https://i.imgur.com/BqxKFvc.png)

```
y_pred = (y_pred > 0.5)
```
In this step, you've created predictions using the predict method. You have also set the threshold for determining if an employee is likely to leave. In the next step, you will use a _confusion matrix_ to evaluate how well the model performed on the predictions.

## Step 6 — Checking the Confusion Matrix

<!--TODO: Let's tell the reader what a confusion matrix is/does, a definition. Provide some good explanation for this. This step is really short for such a complex topic, and something that is integral to machine learning. It is definitely worth giving some context to this.-->

In this step, you will use a _confusion matrix_ to check the number of correct and wrong predictions.  A confusion matrix also is known as an error matrix a square matrix that reports the number of true positives(tp), false positives(fp), true negatives(tn) and false negatives(fn) of a classifier. True positive is an outcome where the model correctly predicts the positive class. It is also known as sensitivity or recall. A false negative is an outcome where the model correctly predicts the negative class. A false positive is an outcome where the model incorrectly predicts the positive class. And a false negative is an outcome where the model incorrectly predicts the negative class

To achieve this you'll use a confusion matrix that `Scikit-learn` provides.

Now add this code to your notebook cell and run it to import the Scikit-learn confusion matrix:

```
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
```
![Checking the Confusion Matrix](https://i.imgur.com/CbWWQ7U.png)

The confusion matrix below means that your deep learning model made 3305 + 375 correct predictions and 106 + 714 wrong predictions. <!--TODO:  --> You can calculate the accuracy has (3305 + 375) / 4500. The total number of observations in our dataset is 4500. This gives us an accuracy of 81.7%.

```
[secondary_label Output]
array([[3305,  106],
       [ 714,  375]])
```
In this step, you have seen how to evaluate your model using the confusion matrix. In the next step, you will work on making a single prediction using the model that you have developed.


## Step 7 — Making a Single Prediction

In this step, you will use your model to make a single prediction given the details of one employee. You will achieve this by predicting the probability of a single employee leaving the company. You'll pass this employee's features to the predict method. As you did earlier, you'll scale the features as well and convert them to a NumPy array.

Now add this code to the Jupyter Notebook cell and run it to pass the employee's features. <!--TODO "to pass the employee's features"? -->

![Making a Single Prediction](https://i.imgur.com/6QBWjvT.png)
```
new_pred = classifier.predict(sc.transform(np.array([[0.26,0.7 ,3., 238., 6., 0.,0.,0.,0., 0.,0.,0.,0.,0.,1.,0., 0.,1.]])))
```

<!--TODO: Add additional explanation here to reiterate the threshold percentage and the predicted value. -->
You then set a threshold of 50% and check the predicted value. Now add this code to the Jupyter Notebook cell and run it. Adding a threshold of 50% means that where the probability is above 50% then that indicates that an employee will leave the company.

![Setting the prediction threshold](https://i.imgur.com/72OlVVZ.png)

```
new_pred = (new_pred > 0.5)
new_pred
```

You can see in your output that the employee won't leave the company:

```
[secondary_label Output]
array([[False]])
```
You might decide to set a lower or higher threshold for your model. For example, you can set the threshold to be 60% as shown below.
```
new_pred = (new_pred > 0.6)
new_pred
```

<!--TODO: Since this step is very short, let's round out with perhaps telling the reader how they can alter the threshold to affect the type of data they will receive. -->


In this step, you have seen how to make a single prediction given the features of a single employee. In the next step, you will work on improving the accuracy of your model.

## Step 8 — Improving the Model Accuracy

In this step, you will use _K-fold cross-validation_ to improve the accuracy of the model that you built earlier. You notice that if you train the model many times you'll keep getting different results. The accuracies for each training have a high variance. In order to solve this problem, you use the _K-fold cross-validation_. Usually, K is set to 10. In this technique, the model is trained on the first 9 folds and tested on the last fold. This iteration continues until all folds have been used. Each of the iterations gives its own accuracy. The accuracy of the model becomes the average of all these accuracies.

Keras enables you to implement K-fold cross-validation via the `KerasClassifier` wrapper. This wrapper is from Scikit-learn's cross-validation. You'll start by importing the `cross_val_score` cross-validation function and the `KerasClassifier`. To do this, insert and run the following code in your notebook cell:

![Improving  the Model Accuracy](https://i.imgur.com/xUhxj5F.png)

```
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
```

<!--TODO: This paragraph/sentence should be introducing the code, as in "Add this code to do X". The detailed explanation of what is happening should come after the code block.  -->

Now add this code to the Jupyter Notebook cell and run it in order to create the function that will be passed to the `KerasClassifier`.

![Creating the classifier function](https://i.imgur.com/vcQiQlP.png)

```
def make_classifier():
    classifier = Sequential()
    classifier.add(Dense(9, kernel_initializer = "uniform", activation = "relu", input_dim=18))
    classifier.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))
    classifier.compile(optimizer= "adam",loss = "binary_crossentropy",metrics = ["accuracy"])
    return classifier
```

<!--TODO Let's add a detailed explanation for the above code here,  -->

In the above code, you have created a function that will be passed to the `KerasClassifier`. This is one of the arguments it expects. The function is a wrapper of the neural network design that we used earlier. The parameters passed are similar to the ones used earlier. In the function, you first initialize the classifier using `Sequential()`, you then use `dense` to add the input and output layer. Finally, you compile the classifier and return it. 

<!--TODO Here, let's also move the bullets and detail on what is happening to the command after. And introduce the code block with a simple sentence, to tell the reader to add the code and a gist of what the code will do. The detailed explanation comes after. -->

Now add this code to the Jupyter Notebook cell and run it. This line of code passes the function you just built to the `KerasClassifier`.

![setting the KerasClassifier function](https://i.imgur.com/fIktN6V.png)

```
classifier = KerasClassifier(build_fn = make_classifier, batch_size=10, nb_epoch=1)
```

The `KerasClassifier` takes three arguments:
- `build_fn` the function with the neural network design
- `batch_size` the number of samples to be passed via the network in each iteration
- `nb_epoch` the number of epochs the network will run


Next, you apply the cross-validation using Scikit-learn's `cross_val_score`. Add the following code to your notebook cell and run it:

![Applying cross-validation using Scikit-learn](https://i.imgur.com/AicexkN.png)

```
accuracies = cross_val_score(estimator = classifier,X = X_train,y = y_train,cv = 10,n_jobs = -1)
```

This function will give you 10 accuracies since you have specified the number of folds as 10. You, therefore, assign it to the accuracies variable and later use it to compute the mean accuracy. It takes the following arguments:
- `estimator` which is the classifier that we defined above
- `X` the training set features
- `y` the value to be predicted in the training set
- `cv` the number of folds
- `n_jobs` the number of CPUs to use. Specifying it as -1 will make use of all the available CPUs

Now you have applied the cross-validation, you can now compute the mean and variance of the accuracies. To achieve this, insert the following code into your notebook cell and run:

![Computing the mean accuracies](https://i.imgur.com/HKoBP6o.png)

```
mean = accuracies.mean()
mean
```

You'll see output similar to the following:

```
[secondary_label Output]
0.8343617910685696
```
Now add this code to the Jupyter Notebook cell and run it to compute the variance of the accuracies:

![Computing the mean variance](https://i.imgur.com/vCx0w1B.png)

```
variance = accuracies.var()
variance
```
You'll see output similar to the following:

```
[secondary_label Output]
0.0010935021002275425
```
In this step, you've improved your model's accuracy by using K-Fold Cross-Validation. In the next step, you will work on the overfitting problem.

## Step 9 — Adding Drop out Regularization to Fight Over-Fitting

In this step, you will add a layer to your model that will help you fight over-fitting in your model. Predictive models are prone to a problem known as _overfitting_. This is a scenario whereby the model memorizes the results in the training set and isn't able to generalize on data that it hasn't seen.  Overfitting can be observed from having a very high variance on the accuracies. In neural networks, the technique used to fight to overfit is known as _dropout regularization_. This is achieved by adding a `Dropout` layer in our neural network. It has a `rate` parameter that indicates the number of neurons that will be deactivated at each iteration. The process of deactivating them is usually random. In this case, we specify 0.1 as the rate meaning that 1% of the neurons will be deactivated during the training process. The network design remains the same.
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

<!--TODO Let's add a detailed explanation of what is happening here. This is an important concept for the reader to understand, and I think the step would benefit from more detail. -->
In the above code, you have added a Dropout layer between the input and output layer. having set a dropout rate of 0.1 means that during the training process 15 of the neurons will be deactivated so that the classifier doesn't overfit on the training set. After adding the dropout and output layers you have then compiled the classfier as you have done previously. 


In this step, you have added a Dropout layer in the model in order to fight overfitting. In the next step you will work on further improving the model by tuning the parameters you used while creating the model.

## Step 10 — Hyperparameter Tuning


In this step, you will use _Grid Search_ to search for the best parameters for your deep learning model. This will help in improving model accuracy. _Grid Search_ is the technique that is used to experiment with different model parameters in order to obtain the ones that give us the best accuracy. Scikit-learn provides the `GridSearchCV` function to enable this functionality. You will now proceed to modify the `make_classifier`  function in order to try out different parameters.

Now add this code to the Jupyter Notebook cell and run it. You will modify the `make_classifier` function slightly to allow you to test out different optimizer functions. <!--TODO: Let's provide a brief sentence to introduce the code here. -->

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
<!--TODO: Let's provide detailed explanation for the code here. -->

In the code above, you have started by importing `GridSearchCV` that you will use to test out different parameters. You have then made slight changes to the `make_classifier` function so that you can try different optimizers in your classifier. You also have initialized the classifier, added the input and output layer then compiled the classifier. Finally, you have returned the classifier so that you can use later. 


The next step is to define the classifier like you did before <!--In which step? -->. Enter this code into a Jupyter Notebook cell and run it.

![Defining the classifier](https://i.imgur.com/yCUYKrb.png)

```
classifier = KerasClassifier(build_fn = make_classifier)
```
In the above line, you have defined the classifier using the `KerasClassifier` which expects a function through the `build_fn` parameter. You have called the `KerasClassifier` and passed the `make_classifier` function that you created earlier. 

<!--TODO: Even though you have defined a classifier before, this still needs to be explained. Let's add this here after the code block. -->


You will now proceed to set a couple of parameters that you wish to experiment with. Enter this code into a Jupyter Notebook cell and run it.

![Defining Parameters](https://i.imgur.com/xn6LqYG.png)
```
params = {
    'batch_size':[20,35],
    'nb_epoch':[5,10],
    'optimizer':['adam','rmsprop']
}
```
Here you have added different batch sizes, number of epochs, and different types of optimizer functions.


<!--TODO: Introduce what the reader is about to do here, before they add the code. -->

Now you are going to use the different parameters you have defined above to search for the best parameters using the `GridSearchCV` function. Enter this code into a Jupyter Notebook cell and run it.

![Searching for best parammeters](https://i.imgur.com/xn6LqYG.png)

```
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=params,
                           scoring="accuracy",
                           cv=10)6
```

The grid search function expects the following parameters:

- `estimator` which is the classifier that we are using
- `param_grid` the set of parameters that we are going to test.
- `scoring` the metric we are going to use
- `cv` the number of folds to test on


Next, you fit this grid search to your training dataset. Enter this code into a Jupyter Notebook cell and run it.

![Fititing the grid search](https://i.imgur.com/vaX7P2L.png)


<!--TODO: This code insertion created a ValueError for me, after running through the tutorial as instructed --> <!-- There was a typo but I have fixed it -->
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
You can now check the best parameters for our model. You notice that the best batch size is 20, the best number of epochs is 10 and the `adam` optimizer is the best for our model. Enter this code into a Jupyter Notebook cell and run it.

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


<!--TODO Let's round out the step here. This is the end of the tutorial, so a summary of what the reader has achieved in this step is vital. -->
In this step, you have used Grid Search to figure out the best parameters for your classifier. You have seen that the best `batch_size` is 20, the best `optimizer` is the `adam` optimizer and the best number of epochs is 10. You have also obtained the best accuracy for your classifier as being 82%.

## Conclusion
In this tutorial, we have learned how we can use [Keras](https://keras.io) to build an artificial neural network that predicts the probability that an employee will leave a company. We combined our previous knowledge in machine learning using `Scikit-Learn` to achieve this. There are a couple of things you can do to keep improving this model:
- Try different activation functions
- Use different optimizer functions
- Experiment with a different number of folds

For more practice, you can try a different dataset and see the results you obtain.
