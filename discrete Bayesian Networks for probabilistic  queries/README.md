## Implementing discrete Bayesian Networks to answer probabilistic queries 

The programs were written in Python and it should do the following
to be able to answer probabilistic queries:
1. Read the datasets (in CSV format) available via Blackboard. 
2. Read from a configuration file a predefined structure of each of your Bayesian 
networks or use randomly generated structures.
3. Learn the parameters of the Bayesian networks (one network per dataset) using 
Maximum Likelihood Estimation for example. 
4. Answer probabilistic queries using one of the algorithms provided in the module.
I applied an exact inference by enumeration to calculate the probability
for the discrete Heart and Stroke Disease data sets. Regarding
the structure of the network, for simplicity, I implemented the
Naive Bayes. In the following steps, I declare my solution:
1- To create the CPT table (conditional probability table)
for each node in the network, I wrote a program named
”parameter-learning.py”(), which applies Maximum Likelihood
estimation to create CPT tables. The program gets data from
a training CSV file (A comma-separated values file) and goes
through rows to count matched situations. The program’s input
is a learning-param list (edge of the network’s structure) and
the output is a dictionary of probabilities. The function ”create-CPT” prints a modified format for putting in the config file. If
a node has no parent, it will find possible values for the node
and count compatible samples. If the node has parents, it will
produce all possible combinations of values and find matched
samples. A Laplacian smoothing is added to the probability
formula to avoid zero probability. I used two functions ”read-data” and ”update-variable-key-values” from the workshop’s
materials.
2- I Completed the config file based on the CPT tables(”config-heart-NaiveBayes.txt” and ”config-stroke-NaiveBayes.txt”).
3- I used an exact inference by enumeration algorithm
”BayesNetExactInference.py” [1]. In inference by enumeration, a conditional probability is computed by summing terms
from the full joint distribution” [2]. Therefore, the program
adds all hidden variables of the network to the joint probability
(each child conditional to its parent), loop over all possible
value and count, then calculate probabilities and multiply all
the probabilities. The results are in table 1.
