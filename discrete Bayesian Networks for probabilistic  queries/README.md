## Implementing discrete Bayesian Networks to answer probabilistic queries 

The programs were written in Python and it should do the following
to be able to answer probabilistic queries:<br/>
1. Read the datasets from the CSV format file. <br/>
2. Read the structure and probability of a Bayesian networks use from a configuration file.<br/>
3. Learn the parameters of the Bayesian networks using Maximum Likelihood Estimation for example. <br/>
4. Answer probabilistic queries.<br/>
I applied an exact inference by enumeration to calculate the probability
for the discrete Heart and Stroke Disease data sets. Regarding
the structure of the network, for simplicity, I implemented the
Naive Bayes. In the following steps, I declare my solution:<br/>
# Read from a configuration file a predefined structure of Bayesian networks
1- To create the CPT table (conditional probability table)
for each node in the network, I wrote a program named  
”parameter-learning.py”(GOTO [parameter-learning link](https://github.com/Afsaneh-Karami/Artificial-intelligence/blob/main/discrete%20Bayesian%20Networks%20for%20probabilistic%20%20queries/paremeter_learning.py)), which applies Maximum Likelihood
estimation to create CPT tables. The program gets data from
a training CSV file (A comma-separated values file) and goes
through rows to count matched situations. The program’s input
is a learning-param list (edge of the network’s structure) and
the output is a dictionary of probabilities. The function ”create-CPT” prints a modified format for putting in the config file. If
a node has no parent, it will find possible values for the node
and count compatible samples. If the node has parents, it will
produce all possible combinations of values and find matched
samples. A Laplacian smoothing is added to the probability
formula to avoid zero probability. 
2- I Completed the config file based on the CPT tables(”config-heart-NaiveBayes.txt” and ”config-stroke-NaiveBayes.txt”).
3- I used an exact inference by enumeration algorithm
”BayesNetExactInference.py” [1]. In inference by enumeration, a conditional probability is computed by summing terms
from the full joint distribution” [2]. Therefore, the program
adds all hidden variables of the network to the joint probability
(each child conditional to its parent), loop over all possible
value and count, then calculate probabilities and multiply all
the probabilities. The results are in table 1.
