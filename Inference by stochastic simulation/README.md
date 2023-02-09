# Inference by the stochastic simulation
I used approximate inference to calculate the conditional probability by stochastic simulation.
Increasing the number of samples causes the
distribution used for probability calculation to become more
realistic, so approximate probability converges to the exact
probability. I obtained the best value for the number of
sampling by trial and error when approximate probability
becomes nearly fixed. <br/>
Three methods were used for random sampling:<br/>
1-Rejection Sampling <br/>
2-Likelihood Weighting Sampling <br/>
3- Gibbs Sampling <br/>
## Rejection Sampling 
I used ”BayesNetApproxInference-reject.py” (GOTO [Reject link](https://github.com/Afsaneh-Karami/Artificial-intelligence/blob/main/Inference%20by%20stochastic%20simulation/BayesNetApproxInference_reject.py)) for Rejection sampling. It takes the query, separates the child node, and starts sampling on all others. The algorithm ignored samples
whose random value of evidence is not equal to its value in the
query. Rejection sampling is better than prior sampling, there
are samples associated with joint probability, but the sampling
process is time-consuming. The Likelihood Weighting and
Gibbs methods are more efficient ways because they decrease
the time for the sampling process.
## Likelihood Weighting Sampling
I wrote a program named ”BayesNetApproxInference-weight.py” (GOTO [weight sampling link](https://github.com/Afsaneh-Karami/Artificial-intelligence/blob/main/Inference%20by%20stochastic%20simulation/BayesNetApproxInference-weight.py)) for the Likelihood Weighting method. The algorithm samples non-evidence nodes, for the evidence node puts
the desired value without sampling. It is efficient because all
samples are compatible with the evidence. The parameter w
multiplies in the sample to compensate for this assumption.
At the start point of the ”prior-sample-weight” function, the
value of w is one, every time it reaches an evidence variable,
the ”change-w” function changes the amount of w based on
the conditional probability in the CPT table by multiplying
w in P(evidence—parent). The ”change-w” function finds the
sampled value of parents for each evidence and creates CPT.
The probability formula is to sum the w of the desired sample
and divide it by the sum of all w. Based on my experience, It
is better that the algorithm first sample nodes with no parent.
The Likelihood Weighting sampling affects nodes under the
evidence node by w. If the evidence nodes are at the bottom
of the network’s structure or the conditional probability is an
infrequent event, the Likelihood Weighting method will be
inefficient.
## Gibbs Sampling
I wrote the ”BayesNetApproxInference-Gibbs.py” program (GOTO [Gibbs sampling link](https://github.com/Afsaneh-Karami/Artificial-intelligence/blob/main/Inference%20by%20stochastic%20simulation/BayesNetApproxInference_Gibbs.py))
for the Gibbs sampling. The function ”del-evidence-sampling”
deletes the evidence nodes from the variables’ list for sampling. The ”prior-sample” function samples unevidenced nodes
based on the CPT table and utilizes the evidence variables
as observed values (fix the evidence). The algorithm samples
one random variable at a time conditioned to others, and the
”Markov-blanket-probability” function gives the probability
distribution used for sampling. The function ”Markov-blanketmem” obtains the Markov Blanket of the node, which is its
parents, and its children, and parents of children. Then, the
”Markov-blanket-probability” function calculates the probability which is a multiplied combination of the probability of the
node conditional to its parent, and the probabilities of child
of the node conditional to the parents. This program repeats
sampling for desired times, at each time just one variable
sample, and other variables are the same as the prior sample.<br/>
In this method, samples come from the correct distribution,
and upstream and downstream variables are conditioned based
on evidence nodes, which helps to get the desired distribution
faster than likelihood weighting.
I compared these three methods by calculating the probability
of task 1a for 2000 samples in the folloeing table. The Gibbs sampling
gives a nearer prediction to the exact inference.
<p align="center">
<img width="482" alt="Screenshot 2023-02-09 103910" src="https://user-images.githubusercontent.com/78735911/217789533-532169aa-742c-49ef-9d33-38ff5daa31b1.png">
</p>


