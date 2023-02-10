# Structure learning algorithm
In this part, I tried to create the best Bayesian Network by
PC-stable and Hill Climbing algorithm. The performance of
Naive Bayes and Bayesian Networks are compared based on
the BIC (Bayesian information criterion). I used stroke data
sets for this part.<br/>
1- I used PC-stable in the program ”make-skeleton-PCstable.py” (GOTO [PCstable link](https://github.com/Afsaneh-Karami/Artificial-intelligence/blob/main/structure%20%20learning%20algorithm/make_skeleton_PCstable.py)) to draw the skeleton of the network based on
the following picture. 
<p align="center">
<img width="407" alt="Screenshot 2023-02-10 004921" src="https://user-images.githubusercontent.com/78735911/217972259-061c6268-7128-46a8-9c88-5ff98a1983a0.png">
</p>

The code at level zero combines all the combinations of two variables without any
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
files.<br/>
2- The result of the skeleton program was applied in the
”CycleDetector.py” file (GOTO [CycleDetector link](https://github.com/Afsaneh-Karami/Artificial-intelligence/blob/main/structure%20%20learning%20algorithm/CycleDetector.py)) as input to find cycles in the graph. The program
checked the existence of any cycles in the graph and removed
the last edge in the detected cycle to create an acyclic graph
and then recalled the function for the next possible cycle.
Even though it is not a good solution to delete the last edge, I
assume that to go ahead with my project. The program output
is a list of edges of the Bayesian network acyclic graph.<br/>
3- The output of step two goes to the ”making-structure.py”
program (GOTO [making-structure link](https://github.com/Afsaneh-Karami/Artificial-intelligence/blob/main/structure%20%20learning%20algorithm/making_structure.py)) to make the structure for the config file.<br/>
4- I created the config file(config-stroke-skeleton.txt) (GOTO [config link](https://github.com/Afsaneh-Karami/Artificial-intelligence/blob/main/structure%20%20learning%20algorithm/config-stroke%20-%20skeleton.zip)). <br/>
5- I applied ”BayesNetexactInference-Gscore.py” (GOTO [Gscore link](https://github.com/Afsaneh-Karami/Artificial-intelligence/blob/main/structure%20%20learning%20algorithm/BayesNetexactInference_Gscore.py)) to calculate
BIC and compare the result of naive bays with the Bayesian
network. This program get the CPT table from the config file and add the ”prob”
function that can calculate the conditional probability for
several parents. I evaluated two structures by the data sets
”stroke-data-discretized-train.csv”. 
The result of table below showed that in terms of Log-Likelihood,
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
In this way training and inferencing time decrease.<br/>

<p align="center">
<img width="455" alt="Screenshot 2023-02-10 010245" src="https://user-images.githubusercontent.com/78735911/217973941-a8eb45a2-0190-4565-952f-08a19b16cb07.png">
<p/>


