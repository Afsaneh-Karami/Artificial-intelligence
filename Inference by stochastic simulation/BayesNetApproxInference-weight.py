
import sys
import random
import BayesNetUtil as bnu
from BayesNetReader import BayesNetReader


class BayesNetApproxInference(BayesNetReader):
    query = {}
    prob_dist = {}
    seeds = {}
    num_samples = None

    def __init__(self):
        if len(sys.argv) != 4:
            print("USAGE> BayesNetApproxInference-weight.py [your_config_file.txt] [query] [num_samples]")
            print("EXAMPLE> BayesNetApproxInference-weight.py \ config-heart-NaiveBayes.txt \P(target|sex=0,cp=3)\ 2000")
        else:
##            file_name = "config-heart-NaiveBayes.txt"
##            prob_query ="P(target|sex=0,cp=3)"
##            self.num_samples = int(2000)
            file_name = sys.argv[1]
            prob_query =sys.argv[2]
            self.num_samples = int(sys.argv[3])
            super().__init__(file_name)
            self.query = bnu.tokenise_query(prob_query)
            self.prob_dist = self.weighting_sampling()
            print("probability_distribution="+str(self.prob_dist))

    def weighting_sampling(self):
        print("\nSTARTING rejection sampling...")
        query_variable = self.query["query_var"]
        evidence = self.query["evidence"]
        C = {}

        # initialise vector of counts
        for value in self.bn["rv_key_values"][query_variable]:
            value = value.split("|")[0]
            C[value] = 0
            

        # loop to increase counts when the sampled vector consistent w/evidence
        for i in range(0, self.num_samples):
            X,w = self.prior_sample_weight()
            value_to_increase = X[query_variable]
            C[value_to_increase] += w

        return bnu.normalise(C)

    def prior_sample_weight(self):
        X = {}
        w=1
        sampled_var_values = {}
        for variable in self.bn["random_variables"]:
            if variable in self.query["evidence"]:
                new_w=self.change_w(variable,X)
                w=w*new_w
                continue
            else:
                X[variable] = self.get_sampled_value(variable, sampled_var_values)
                sampled_var_values[variable] = X[variable]
        return X,w

    def change_w(self,var,X):
        #self.query["evidence"]
        w_prob={}
        parent_string_value=""
        parent_sample={}
        parents=bnu.get_parents(var, self.bn)
        if parents is None:
            cpt="CPT("+var+")"
            var_value=self.query["evidence"][var]
            p=self.bn[cpt][var_value]
        elif len(parents.split(","))==1:
            cpt="CPT("+var+"|"+parents+")"
            var_value=self.query["evidence"][var]+"|"+X[parents]
            p=self.bn[cpt][var_value]
            
        else:
            parents=parents.split(",")
            for parent in parents:
                if parent in self.query["evidence"]:
                    parent_var=self.query["evidence"][parent]
                    parent_string_value=parent_string_value+var+"|"+parent_var
                    cpt=cpt+"CPT("+var+")|"+parent
                else:
                    parent_sample_var=self.get_sampled_value(parent,parent_sample)
                    parent_string_value=parent_string_value+parent_sample_var
                    cpt=cpt+"CPT("+var+")|"+parent
            p=self.bn[cpt][parent_string]
        return p
    
    def get_sampled_value(self, V, sampled):
        # get the conditional probability distribution (cpt) of variable V
        parents = bnu.get_parents(V, self.bn)
        cpt = {}
        prob_mass = 0

        # generate a cumulative distribution for random variable V
        if parents is None:
            for value, probability in self.bn["CPT("+V+")"].items():
                prob_mass += probability
                cpt[value] = prob_mass

        else:
            for v in bnu.get_domain_values(V, self.bn):
                p = bnu.get_probability_given_parents(V, v, sampled, self.bn)
                prob_mass += p
                cpt[v] = prob_mass

        # check that the cpt sums to 1 (or almost)
        if prob_mass < 0.999 and prob_mass > 1:
            print("ERROR: CPT=%s does not sum to 1" % (cpt))
            exit(0)

        return self.sampling_from_cumulative_distribution(cpt)

    def sampling_from_cumulative_distribution(self, cumulative):
        random_number = random.random()
        for value, probability in cumulative.items():
            if random_number <= probability:
                random_number = random.random()
                return value.split("|")[0]

        print("ERROR couldn't do sampling from:")
        print("cumulative_dist="+str(cumulative))
        exit(0)



BayesNetApproxInference()
