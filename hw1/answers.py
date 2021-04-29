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
4. NEED TO RECHECK

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
classification and harm the its performance.

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

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
