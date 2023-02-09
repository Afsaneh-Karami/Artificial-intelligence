
import sys
import math
import time
import random
import numpy as np
import BayesNetUtil as bnu
from sklearn import metrics
from BayesNetReader import BayesNetReader


class BayesNetApproxInference(BayesNetReader):
    query = {}
    prob_dist = {}
    seeds = {}
    num_samples = None
    num_data_instances = 0

    def __init__(self):
        if len(sys.argv) != 4:
            print("USAGE> BayesNetApproxInference_weight_ test.py [your_config_file.txt] [your_test_file.csv] [num_samples]")
            print("EXAMPLE> BayesNetApproxInference_weight_ test.py \ config-heart-NaiveBayes.txt \heart-data-discretized-test.csv\ 1000")
        else:
            
            self.predictions={}
##            file_name = "config-heart-NaiveBayes.txt"
            file_name = sys.argv[1]
            super().__init__(file_name)
            self.inference_time = time.time()
##            file_name_test="heart-data-discretized-test.csv"
##            self.num_samples = int(1000)
            file_name_test=sys.argv[2]
            self.num_samples = int(sys.argv[3])
            self.read_data_test(file_name_test)
            self.query_variable=self.rand_vars[-1]
            for instance in range(0,len(self.rv_all_values)):
                 prob_query,self.query_evidence =self.creat_query(self.rv_all_values[instance])
                 self.prob_dist = self.weighting_sampling()
                 self.predictions[instance]=self.prob_dist
                 #print("for instance "+str(self.rv_all_values.index(instance))+" probability_distribution="+str(self.prob_dist))
            self.inference_time = time.time() - self.inference_time
            self.compute_performance()
    def read_data_test(self, file_name_test):
        print("\nREADING data file %s..." % (file_name_test))
        print("---------------------------------------")

        self.rand_vars = []
        self.rv_key_values = {}
        self.rv_all_values = []

        with open(file_name_test) as csv_file:
            for line in csv_file:
                line = line.strip()
                if len(self.rand_vars) == 0:
                    self.rand_vars = line.split(',')
                    for variable in self.rand_vars:
                        self.rv_key_values[variable] = []
                else:
                    values = line.split(',')
                    self.rv_all_values.append(values)
                    self.update_variable_key_values(values)
                    self.num_data_instances += 1
        self.predictor_variable = self.rand_vars[len(self.rand_vars)-1]
        print("RANDOM VARIABLES=%s" % (self.rand_vars))
        print("VARIABLE KEY VALUES=%s" % (self.rv_key_values))
        print("VARIABLE VALUES (first 10)=%s" % (self.rv_all_values[:10]))
        print("PREDICTOR VARIABLE=%s" % (self.predictor_variable))
        print("|data instances|=%d" % (self.num_data_instances))
        
    def update_variable_key_values(self, values):
        for i in range(0, len(self.rand_vars)):
            variable = self.rand_vars[i]
            key_values = self.rv_key_values[variable]
            value_in_focus = values[i]
            if value_in_focus not in key_values:
                self.rv_key_values[variable].append(value_in_focus)

    def creat_query(self,instance):
        evidence={}
        query_evidence=""
        for var_index in range(0,len(self.rand_vars)-1):
            var_value=self.rand_vars[var_index]
            var_value_instance=instance[var_index]
            evidence[var_value]=var_value_instance
            query_evidence=query_evidence+var_value+"="+var_value_instance+","
        query_evidence_modify=query_evidence[0:len(query_evidence)-1]
        query="p("+self.predictor_variable+"|"+query_evidence_modify+")"
        return query,evidence

    def weighting_sampling(self):
        C = {}
        # initialise vector of counts
        for value in self.bn["rv_key_values"][self.query_variable]:
            value = value.split("|")[0]
            C[value] = 0
        # loop to increase counts when the sampled vector consistent w/evidence
        for i in range(0, self.num_samples):
            X,w = self.prior_sample_weight()
            value_to_increase = X[self.query_variable]
            C[value_to_increase] += w

        return bnu.normalise(C)

    def prior_sample_weight(self):
        X = {}
        w=1
        sampled_var_values = {}
        for variable in self.bn["random_variables"]:
            if variable in self.query_evidence:
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
            var_value=self.query_evidence[var]
            p=self.bn[cpt][var_value]
        elif len(parents.split(","))==1:
            cpt="CPT("+var+"|"+parents+")"
            var_value=self.query_evidence[var]+"|"+X[parents]
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

    def compute_performance(self):
        Y_true = []
        Y_pred = []
        Y_prob = []

        # obtain vectors of categorical and probabilistic predictions
        for i in range(0, len(self.rv_all_values)):
            target_value = self.rv_all_values[i][len(self.rand_vars)-1]
            if target_value == 'yes': Y_true.append(1)
            elif target_value == 'no': Y_true.append(0)
            elif target_value == '1': Y_true.append(1)
            elif target_value == '0': Y_true.append(0)
            
            predicted_output = self.predictions[i][target_value]
            Y_prob.append(predicted_output)
            best_key = max(self.predictions[i], key=self.predictions[i].get)
            if best_key == 'yes': Y_pred.append(1)
            elif best_key == 'no': Y_pred.append(0)
            elif best_key == '1': Y_pred.append(1)
            elif best_key == '0': Y_pred.append(0)
        
        P = np.asarray(Y_true)+0.00001 # constant to avoid NAN in KL divergence
        Q = np.asarray(Y_prob)+0.00001 # constant to avoid NAN in KL divergence

        # calculate metrics: accuracy, auc, brief, kl, training/inference times
        acc = metrics.balanced_accuracy_score(Y_true, Y_pred)
        fpr, tpr, _ = metrics.roc_curve(Y_true, Y_prob, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        brier = metrics.brier_score_loss(Y_true, Y_prob)
        kl_div = np.sum(P*np.log(P/Q))
        print("PERFORMANCE:")
        print("Balanced Accuracy="+str(acc))
        print("Area Under Curve="+str(auc))
        print("Brier Score="+str(brier))
        print("KL Divergence="+str(kl_div))
        #print("Training Time="+str(self.training_time)+" secs.")
        print("Inference Time="+str(self.inference_time)+" secs.")



BayesNetApproxInference()
