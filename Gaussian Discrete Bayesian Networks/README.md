# Gaussian Discrete Bayesian Networks
<p align="justify">
This code support conditional Gaussian Bayesian Nets – combining discrete and continuous random variables. I compare the
performance of gaussian discrete bayesian networks(using the original datasets without missing values) against discrete Bayesian 
Networks (using the discretised datasets) by metrics such as Log Likelihood (LL) and BIC (Bayesian Information Criterion). <br/>
I also applied gussian sampling for inference by stochastic simulation and compare the result with weighting sampling with discrete datd.<br/>
I used the probability density function because data sets are continuous. I used Gaussian (normal distribution) to calculate probability density and Naive Bayes to create the structure. The solution for this task includes:<br/>
</p>

1- I used the program ”PDF-Generator.py” (GOTO [PDF generator link](https://github.com/Afsaneh-Karami/Artificial-intelligence/blob/main/Gaussian%20Discrete%20Bayesian%20Networks/PDF_Generator.py)) to make a
probability density for each node which represents the probability density for each node related to its parent. The program calculates the mean and standard deviation for nodes with no parent calculates based on the node values. If the node has some parents, the code applies a linear Gaussian model as ridge regression (a regression method) to find probability density. At first, the variables and structure of the network wrote in config files (”config-heart-NaiveBayes-Gaussain.txt”
and ”config-stroke-NaiveBayes-Gaussain.txt”(GOTO [config link](https://github.com/Afsaneh-Karami/Artificial-intelligence/blob/main/Gaussian%20Discrete%20Bayesian%20Networks/config-heart-NaiveBayes-Gussain.txt))). Then, the program runs to create probability density.<br/>
2- I assessed the performance of Gaussian Naive Bayes for two data sets, Heart and Stroke, by the program ”NB-Classifier-v3.py” (GOTO [NB-Classifier link](https://github.com/Afsaneh-Karami/Artificial-intelligence/blob/main/Gaussian%20Discrete%20Bayesian%20Networks/NB_Classifier_v3.py)) . Also, I compared the performance of Naive Bays for discrete and continuous variables and presented the results
in the following table. For some metrics, Gaussian performs better like AUC which means it can better distinguish between classes, Brier score which means a better model, and KL divergence which means more similar probability distribution. But it takes more time to classify with Gaussian Bayesian. Overall the continuous variable gives more accurate results.

<p align="center">
<img width="406" alt="Screenshot 2023-02-10 091233" src="https://user-images.githubusercontent.com/78735911/218051349-95df3bb4-d193-46e2-b979-cfbd4610a283.png">
</p>

## Gaussian sampling and comparing with weighting sampling:
I wrote a program ”GussianBayesNetApproxInference.py” (GOTO [GussianBayesNetApproxInference link](https://github.com/Afsaneh-Karami/Artificial-intelligence/blob/main/Gaussian%20Discrete%20Bayesian%20Networks/GussianBayesNetApproxInference.py))
for the approximate inference methods with Gaussian sampling. It works more like Likelihood Weighting sampling.
This program takes the probability density from the config
file and evaluates the sampling performance accuracy on the
test data. The ”creat-query” function creates a query based on
each row of test data sets. The ”prior-sample-weight” function
decides to sample the variable or change the w (the same
as Likelihood Weighting sampling). The ”change-w” function
modifies the w based on evidence values. The ”get-sampled-value” function produces the probability density and mean and
standard deviation used for sampling. The mean and standard
deviation are calculated like before (first part of section 1c).
The ”sampling-from-cumulative-distribution” function creates
a random value based on the Gaussian distribution of the node.
I compared the accuracy of Gaussian sampling for continuous
variables with Likelihood Weighting sampling for discrete
variables for Heart data sets in the following table.<br/>
<p align="center">
<img width="332" alt="Screenshot 2023-02-10 091929" src="https://user-images.githubusercontent.com/78735911/218052967-8ff06516-6c6e-4669-991f-2e86daed744b.png">
</p>


