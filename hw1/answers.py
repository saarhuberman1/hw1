r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1. False, the in sample error is calculated on the training set. we use the test set to estimate the out of sample error.
2. False, for example if you do not have enough diversity in your training data, e.g you missing part of labels.
for example in cifar 10 all the labeled 'dog' examples are only in the test set, than the model can not learn that
 label.
3. True, the test set is needed to evaluate how the model generelizes to unseen data for estimating the out of sample 
performance. the cross validation is used to choose the the hyper parameter of the model, meaning it is part of 
the final model construction. if the cross validation will use the test data the results will not be reliable.
4. False, The validation set performance is used to tune the hyper parameters of the model so the choice of the 
final model is effected by the validation set. 
the generalization error is a measure of how accurately the model is able to predict outcome values for previously 
unseen data because of that the performance of the model on the validation set 
can not be a proxy to the generalization error.

"""

part1_q2 = r"""
**Your answer:**

my friends approach is not justified. tuning the hyper parameter lambda using the test set will fit the data 
classifier to the test data. so the testing set will not examine well the generalization of the model.
the appropriate way is to split the data to 3 disjoint sets. training, validation and test sets. the models parameter 
will be learn using the train set, the hyper parameters such as lambda will be tune using the validation set and the 
model generalization will be examine on the test set.

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

Increasing K lead to improve in generalization for unseen data up to a certain point and than cause a deterioration 
in the performance. the results for k=3 are better than for k=1, but for k > 3, the results are in downward trend.
this can be explained by that the averaging 3 nearest points  can help generalize and prevent noise and anomalies to 
affect the classification. but for larger k's you use more distant data point that are not relevant for the 
classification and harm its performance.
If k=1, we decide what will be the label of every new data-point based on its (one) closest neighbor. 
This is very unstable as any wrong label in the training data will affect the clustering of all the data-points that
it's their closest neighbors from the test set.
On the other hand, if k=number_of_training_samples we decide the label of any new data-point to be the same as the 
label that is most frequent in our training set.

"""

part2_q2 = r"""
**Your answer:**

1. using K-fold CV is better than training on the entire training set with various models and selecting the best model 
with respect to the training set, those because evaluating the different models on the same data will not choose 
the model that best generalizes, instead it will choose the model that best fit the training data. 
for the knn, when using k=1 if there are no discrepancy in the data the model will fit perfectly on the training data, 
while k=3 will better generalize to unseen data.
2. using K-fold CV is better than training on the entire training set with various models and selecting the best model 
   with respect to the test set. when choosing the best model(or tuning hyper parameters) with the test set you cannot 
   reliably evaluate the generalization of the model with the test set, since it was a part of its selecting. that
   in contrast of using the K-fold CV for choosing the model.
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

TODO: do they mean that delta doesn't have to be >0 or that delta can be any value >0 ?

"""

part3_q2 = r"""
**Your answer:**

1. Each data point is a 785-d vector (of 784 pixels + 1 for the bias). 
The SVM model tries to find a way to divide the 785 space into regions (subspaces)
using hyper-planes such that each region corresponds with a different class 
in the way that the projection of each vector to the corresponding subspace of the true label has the highest value
between all other projections of the same vector to the rest of the subspaces.
We can see some classification errors for digits that look like other digits for example confusion between 7 and 9. 
As the vectors representing their pixels might be close.

2. In the SVM model, the decision boundaries are hyperplanes while in KNN the decision boundaries are non-linear. 
The SVM model looks at all the training data and tries to fit the best hyperplanes based on all the available information 
while in KNN, the decision is based only on the closest K samples from the training set. 
The models are similar because they both aim to divide the space into regions where each region corresponds to 
a different class. 

"""

part3_q3 = r"""
**Your answer:**


1. The learning rate is good. 
If the learning was too low, the learning process will take alot of epochs to converge (and in our case it converges fast)
If the learning was too high, the learning process will diverge as the weight marix will be updated with large steps and might miss the minima


2. Our model is slightly over fitting as we can see it performs well on both train and test set however, 
the training set accuracy is slightly higher than the test-set accuracy. 
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**

The ideal pattern to see in a residual plot is a straight line at y=0 as this means the 
y_pred-y_real=0 for every sample, meaning that the model prediction is perfect. 
If we look at the residuals graph for the top 5 features, we see that some samples fall far from the y=0 line 
(and they form what appears to be a straight line - TODO what can we learn from this??), 
meaning the prediction was far from the true label for these samples. However, looking at the final residual plot, 
we see that the points are much closer to the y=0 line, meaning the model fitted the data much better.
  
   

"""

part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q3 = r"""
**Your answer:**

1. Using np.logspace to chose values of $\lambda$ allows us to check the model performance for different orders 
of magnitude of $\lambds$ as close values of $\lambda$ will probably produce similar results. Therefore, this allows us 
to check a wider range of values with less iterations.

2. The model was fitted 180 times: 3 folds, 3 degrees (1,2 and 3) and 20 different value for $\lambda$.
 $3 \times 3 \times 20 = 180$ in total 


"""

# ==============
