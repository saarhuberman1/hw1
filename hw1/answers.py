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

the choice of $\Delta$ is arbitrary for $\Delta$ > 0 because choosing when a new $\Delta'$, we can find an appropriate 
$W' = \frac{\Delta'}{\Delta} \cdot W$ and $\lambda' = \frac{\Delta}{\Delta'}\cdot \lambda$:

$$ L_{i}(\mat{W'}) =  \sum_{j \neq y_i} \max\left(0, \Delta' + \vectr{{w'}_j} \vec{x_i} - \vectr{{w'}_{y_i}} \vec{x_i}\right) = 
\sum_{j \neq y_i} \max\left(0, \Delta' + \frac{\Delta'}{\Delta} \cdot \vectr{w_j} \vec{x_i} - 
\frac{\Delta'}{\Delta} \cdot \vectr{w_{y_i}} \vec{x_i}\right) = 
\frac{\Delta'}{\Delta} \cdot \sum_{j \neq y_i} \max\left(0, \Delta + \vectr{w_j} \vec{x_i} - \vectr{w_{y_i}} \vec{x_i}\right)
$$

$$ \frac{\lambda'}{2} \norm{\mat{W'}}^2 = 
\frac{\Delta}{\Delta'}\cdot \frac{\lambda}{2} ({\frac{\Delta'}{\Delta}})^2 \cdot \norm{\mat{W}}^2 =
\frac{\Delta'}{\Delta} \cdot \frac{\lambda}{2} \norm{\mat{W}}^2
$$

$$ \Rightarrow L(\mat{W'}) = \frac{\Delta'}{\Delta} \cdot L(\mat{W'})
$$

for every $\Delta$ we can find weights and tune a hyper parameter \lambda, which result in the same loss up to scaling. 
this weights will be the same weights up to a constant factor.


"""
# \frac{\Delta}{\Delta'}\cdot \frac{\lambda}{2} ({\frac{\Delta'}{\Delta}})^2 \cdot \norm{\mat{W}}^2 =
# \frac{\Delta'}{\Delta} \cdot \frac{\lambda}{2} \norm{\mat{W}}^2
part3_q2 = r"""
**Your answer:**

1. The linear model learn representation for each class that is encoded in the weights matrix. 
you can see in the weights visualization that for each class the weights look like the class represented number itself.
in inference time the model do matrix multiplication between the classes weights matrix and the requested example, 
resulting in correlation vector of this example and each of the classes, when choosing the best correlated class. 
the miss classify examples look similar to the learned representations of similar looking number classes, 
and have more correlated areas than the true label class.

2. this interpretation is different from the knn approach. in this case the model try to find a global 
representation for each class and divide the feature space by an hyperplane. the knn model divide the feature space 
to small regions infused by the metric and the training example, when each region correspond to a single class.

"""

part3_q3 = r"""
**Your answer:**
1. Based on the graph of the training loss the learning rate we choose was good. if the learning rate was too low the 
training loss will descend very slowly and will still be in be in negative slope. if the learning rate was too high the 
training loss graph will be very spiky and will not converge to 0.

2. Our model is slightly over fitted as we can see it performs well on both train and test set, however 
the training set accuracy is slightly higher than the test-set accuracy. 

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**

The ideal pattern to see in a residual plot is a straight line at y=0 as this means the 
y_pred-y_real=0 for every sample, meaning that the model prediction is perfect.

If we look at the residuals graph for the top 5 features, big portion of the samples are in a small bounded margin from 
the y=0 line, but we see that some samples fall far from that are meaning the prediction was far from the ground truth 
value for these samples. However, looking at the final residual plot, the points are more centered to the zero line, 
and most of the points are bounded at that margin, while the points that are outside are much closer to it meaning the 
model fitted the data much better.

"""

part4_q2 = r"""
**Your answer:**

adding more non-linear features to the data lift it to higher dimension and  enrich it with more information.

1. this is still a linear regression model, as it calculated the a matrix multiplication between the features and the 
weights, although the features that we used for the regression have a non linear connection the original data.
2. in theory we can can fit an approximation of any non linear function in this approach, if we know the necessary 
features.
3. the decision boundary will remain an hyper plane in the feature dimensions, but in the original data domain the 
decision boundary will not be an hyper plane, instead it will be much more complex boundary.

"""

part4_q3 = r"""
**Your answer:**

1. Using np.logspace to chose values of $\lambda$ allows us to check the model performance for different orders 
of magnitude of $\lambda$ as close values of $\lambda$ will probably produce similar results. Therefore, this allows us 
to check a wider range of values with less iterations.
2. The model was fitted 180 times: 3 folds, 3 degrees (1,2 and 3) and 20 different value for $\lambda$.
 $3 \times 3 \times 20 = 180$ in total 

"""

# ==============
