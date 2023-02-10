# Structure learning algorithm
In this part, I tried to create the best Bayesian Network by
PC-stable and Hill Climbing algorithm. The performance of
Naive Bayes and Bayesian Networks are compared based on
the BIC (Bayesian information criterion). I used stroke data
sets for this part.<br/>
1- I used PC-stable in the program ”make-skeleton-PCstable.py” (GOTO [Reject link](https://github.com/Afsaneh-Karami/Artificial-intelligence/blob/main/Inference%20by%20stochastic%20simulation/BayesNetApproxInference_reject.py)) to draw the skeleton of the network based on
the picture from lecture week 3 slide 7. The code at level zero
combines all the combinations of two variables without any
parent and makes a list of dependent edges. I implemented
the null hypothesis to evaluate independence between two
nodes based on the data set and create edges between two
nodes with a mark lower than 0.01. Then went through
the edges from level zero and added one possible parent to
it (a possible parent means at least there should be a link
edge between the parent and one of the nodes in the edge).
The algorithm calculated the probability of independence
at level one and removed independent edges. Increase the
number of possible parents until all possible combinations
are checked. The program’s output is a list of edges in the
Bayesian network. About the stroke data sets, the residence
type variable was assessed as independent from all other
variables, so I eliminated it from the test and train data CSV
files.
2- The result of the skeleton program was applied in the
”CycleDetector.py” file as input to find cycles in the graph.
I used the workshop material and changed it. The program
checked the existence of any cycles in the graph and removed
the last edge in the detected cycle to create an acyclic graph
and then recalled the function for the next possible cycle.
Even though it is not a good solution to delete the last edge, I
assume that to go ahead with my project. The program output
is a list of edges of the Bayesian network acyclic graph.
3- The output of step two goes to the ”making-structure.py”
program to make the structure for the config file.
4- I created the config file(config-stroke-skeleton.txt).
5- I applied ”BayesNetexactInference-Gscore.py” to calculate
BIC and compare the result of naive bays with the Bayesian
network. This program is a workshop material that I change
to get the CPT table from the config file and add the ”prob”
function that can calculate the conditional probability for
several parents. I evaluated two structures by the data sets
”stroke-data-discretized-train.csv”.
The result of table 4 showed that in terms of Log-Likelihood,
the Bayesian network is better than Naive Bayes, but it has
a high penalty value that makes the Bayesian Information
Criterion so much. The high value of the penalty means
that the Bayesian network is complicated, and the risk of
overfitting is high. One solution is to check the effect of
each edge on BIC value and eliminate some of them to
increase BIC. As creating the CPT for the Bayesian network
took so much time (about three hours), this solution is so
time-consuming. Another option is to decrease the complexity
of the structure (decrease penalty) by limiting the number of
parents to three, who have the lowest null hypothesis value.
In this way training and inferencing time decrease.
