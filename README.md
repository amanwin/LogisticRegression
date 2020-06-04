# Logistic Regression

**Logistic regression**, which is a **classification model**, i.e. it will help you make predictions in cases where the output is a **categorical variable**. Since logistic regression is the most easily interpretable of all classification models, it is very commonly used in various industries such as banking, healthcare, etc.

## Introduction: Univariate Logistic Regression
In this session, you will learn a few basic concepts related to logistic regression. Broadly speaking, the topics that will be covered in this session are:

* Binary classification
* Sigmoid function
* Likelihood function
* Building a logistic regression model in Python
* Odds and log odds

## Binary Classification
The most common use of logistic regression models is in binary classification problems.

#### Examples of Classification
1. Finance company wants to know whether a customer is default or not.
2. Predicting an email is spam or not.
3. Categorizing email into promotional, personal and official.

A classification problem where we have two possible outputs/outcomes is called as **Binary Classification problem.**
* Examples
    1. Customer default or not.
    2. Spam/ham example.
    3. Categorizing email into promotional, personal and official. This is not a binary classification problem but a multi class classification problem.


[Diabetes Data](dataset/DiabetesExampleData.csv)

Now, recall the graph of the diabetes example. Suppose there is another person, with a blood sugar level of 195, and you do not know whether that person has diabetes or not. What would you do then? Would you classify him/her as a diabetic or as a non-diabetic?

![title](image/e2e2e2.png)

Now, based on the boundary, you may be tempted to declare this person a diabetic, but can you really do that? This person’s sugar level (195 mg/dL) is very close to the threshold (200 mg/dL), below which people are declared as non-diabetic. It is, therefore, quite possible that this person was just a non-diabetic with a slightly high blood sugar level. After all, the data does have people with slightly high sugar levels (220 mg/dL), who are not diabetics.

## Sigmoid Curve
In the last section, you saw what a binary classification problem is, and then you saw an example of a binary classification problem, where a model is trying to predict whether a person has diabetes or not based on his/her blood sugar level. We saw how using a **simple boundary decision method** would not work in this case.

Since the **sigmoid curve** has all the properties you would want — extremely low values in the start, extremely high values in the end, and intermediate values in the middle — it’s a good choice for modelling the value of the **probability of diabetes.**

![title](image/sigmoid-curve.JPG)

This is the sigmoid curve equation:

![title](image/sigmoid-curve-equation.JPG)

So now we have verified, with actual values, that the sigmoid curve actually has the properties we discussed earlier, i.e. extremely low values in the start, extremely high values in the end, and intermediate values in the middle.

However, you may be wondering — why can’t you just fit a straight line here? This would also have the same properties — low values in the start, high ones towards the end, and intermediate ones in the middle.

![title](image/diabetes.png)

The main problem with a straight line is that it is not steep enough. In the sigmoid curve, as you can see, you have low values for a lot of points, then the values rise all of a sudden, after which you have a lot of high values. In a straight line though, the values rise from low to high very uniformly, and hence, the “boundary” region, the one where the probabilities transition from high to low is not present.

## Finding the Best Fit Sigmoid Curve - I

So, in the previous lecture, you saw what a sigmoid function is and why it is a good choice for modelling the probability of a class. Now, in this section, you will learn how you can find the best fit sigmoid curve. In other words, you will learn how to find the combination of β0 and β1 which fits the data best.

By varying the values of β0 and β1, we get different sigmoid curves. Now, based on some function that you have to minimise or maximise, you will get the best fit sigmoid curve.

![title](image/sigmoid-curve-best-fit.JPG)

So, the best fitting combination of β0 and β1 will be the one which maximises the product:

![title](image/likelihood-function.JPG)

## Finding the Best Fit Sigmoid Curve - II
In the previous lecture, you understood what a likelihood function is. To recap, the likelihood function for our data is **(1-P1)(1-P2)(1-P3)(1-P4)(1-P6)(P5)(P7)(P8)(P9)(P10) .** The best fitting sigmoid curve would be the one which maximises the value of this product.

If you had to find β0 and β1 for the best fitting sigmoid curve, you would have to try a lot of combinations, unless you arrive at the one which maximises the likelihood. This is similar to linear regression, where you vary β0 and β1 until you find the combination that minimises the cost function. 
This process, where you vary the betas until you find the best fit curve for the probability of diabetes, is called **logistic regression.**

## Odds and Log Odds
In the previous segment, you saw that by trying different values of β0 and β1 , you can manipulate the shape of the sigmoid curve. At some combination of β0 and β1, the 'likelihood' will be maximised.

###  Logistic Regression in Python
Let's now look at how logistic regression is implemented in python.In python, logistic regression can be implemented using libraries such as SKLearn and statsmodels, though looking at the coefficients and the model summary is easier using statsmodels. 

You can find the optimum values of β0 and β1 using the python code given below. Please download and run the code and observe the values of the coefficients.

Please note that you will study a detailed Python code for logistic regression in the next module. This Python code has been run so as to find the optimum values of β0 and β1 so that we can first proceed with the very important concept of **Odds and Log Odds**.

[Finding Optimum Betas using Python](dataset/Betas+for+Logistic+Regression.ipynb)

The summary of the model is given below:

![title](image/summary.png)

In the summary shown above, 'const' corresponds to β0 and Blood Sugar Level, i.e. 'x1' corresponds to β1. So, β0 = -13.5 and β1 = 0.06.

### Odds and Log Odds
So far, you’ve seen this equation for logistic regression:

![title](image/formula.JPG)

Recall that this equation gives the relationship between P, the probability of diabetes and x, the patient’s blood sugar level.

While the equation is correct, it is not very intuitive. In other words, the relationship between P and x is so complex that it is difficult to understand what kind of trend exists between the two. If you increase x by regular intervals of, say, 11.5, how will that affect the probability? Will it also increase by some regular interval? If not, what will happen?

So, clearly, the relationship between P and x is too complex to see any apparent trends. However, if you convert the equation to a slightly different form, you can achieve a much more intuitive relationship. 

[Note: By default, for this course, if the base of the logarithm is not specified, take it as e. So, log(x) = loge(x)]

![title](image/sigmoid-solving.JPG)

![title](image/sigmoid-interpretation.JPG)

So, now, instead of probability, you have odds and log odds. Clearly, the relationship between them and x is much more intuitive and easy to understand.

![title](image/odds-probablity.JPG)

![title](image/rel-odds-probablity)

![title](image/plot-odds-probablity.JPG)

![title](image/rel-odds.JPG)

![title](image/x-odds.JPG)

So, the relationship between x and probability is not intuitive, while that between x and **odds/log odds** is. This has important implications. Suppose you are discussing sugar levels and the probability they correspond to. While talking about 4 patients with sugar levels of 180, 200, 220 and 240, you will not be able to intuitively understand the relationship between their probabilities (10%, 28%, 58%, 83%). However, if you are talking about the log odds of these 4 patients, you know that their log odds are in a **linearly increasing pattern** (-2.18, -0.92, 0.34, 1.60) and that the odds are in a **multiplicatively increasing pattern** (0.11, 0.40, 1.40, 4.95, increasing by a factor of 3.55).

Hence, many times, it makes more sense to present a logistic regression model’s results in terms of log odds or odds than to talk in terms of probability. This happens especially a lot in industries like finance, banking, etc.

That's the end of this session on univariate logistic regression. You studied logistic regression, specifically, the sigmoid function, which has this equation:

![title](image/formula.JPG)

However, this is not the only form of equation for logistic regression. There is also the probit form and cloglog form of logistic regression.

## Multivariate Logistic Regression - Model Building

### Introduction
Just like when you’re building a model using linear regression, one independent variable might not be enough to capture all the uncertainties of the target variable in logistic regression as well. So in order to make good and accurate predictions, you need multiple variables and that is what we’ll study in this session.

Before starting with multivariate logistic regression, the first question that arises is, “Do you need any extensions while moving from univariate to multivariate logistic regression?” Recall the equation used in the case of univariate logistic regression was:

![title](image/formula.JPG)

The above equation has only one feature variable X, for which the coefficient is β1. Now, if you have multiple features, say n, you can simply extend this equation with ‘n’ feature variables and ‘n’ corresponding coefficients such that the equation now becomes:

![title](image/formula1.JPG)

Recall this extension is similar to what you did while moving from simple to multiple linear regression.

## Multivariate Logistic Regression - Telecom Churn Example
Let's now look at the process of building a logistic regression model in Python.

You will be looking at the telecom churn prediction example. You will use 21 variables related to customer behaviour (such as the monthly bill, internet usage etc.) to predict whether a particular customer will switch to another telecom provider or not (i.e. churn or not).

#### Problem Statment
You have a telecom firm which has collected data of all its customers. The main types of attributes are:
* Demographics (age, gender etc.)
* Services availed (internet packs purchased, special offers taken etc.)
* Expenses (amount of recharge done per month etc.)

Based on all this past information, you want to build a model which will predict whether a particular customer will churn or not, i.e. whether they will switch to a different service provider or not. So the variable of interest, i.e. the target variable here is ‘Churn’ which will tell us whether or not a particular customer has churned. It is a binary variable - 1 means that the customer has churned and 0 means the customer has not churned.

You can download the datasets here:

[Churn Data](dataset/churn_data.csv)

[Internet Data](dataset/internet_data.csv)

[Customer Data](dataset/customer_data.csv)

Also, here’s the data dictionary:

[Telecom Churn Data Dictionary](dataset/TelecomChurnDataDictionary.csv)

You can also download the code file and may follow along. 

[Logistic Regression Telecom Churn Case Study](dataset/Logistic+Regression+-+Telecom+Churn+Case+Study.ipynb)

So, here’s what the data frame churn_data looks like:

![title](image/churn_data.JPG)

Also, here’s the data frame customer_data:

![title](image/customer_data.JPG)

Lastly, here’s the data frame internet_data:

![title](image/internet_data.JPG)

Now, as you can clearly see, the first 5 customer IDs are exactly the same for each of these data frames. Hence, using the column customer ID, you can collate or merge the data into a single data frame. We'll start with that in the next segment.

## Data Cleaning and Preparation - I
Before you jump into the actual model building, you first need to clean and prepare your data. As you might have seen in the last segment, all the useful information is present in three dataframes with ‘Customer ID’ being the common column. So as the first step, you need to merge these three data files so that you have all the useful data combined into a single master dataframe. 