
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
            print("USAGE> GussianBayesNetApproxInference.py [your_config_file.txt] [your_test_file] [num_samples]")
            print("EXAMPLE> GussianBayesNetApproxInference.py config-heart-NaiveBayes-Gussain.txt \heart-data-original-test.csv\ 1000")
        else:
            
            self.predictions={}
            file_name = sys.argv[1]
##            file_name = "config-heart-NaiveBayes-Gussain.txt"
            super().__init__(file_name)
            self.inference_time = time.time()
##            file_name_test="heart-data-original-test.csv"
            file_name_test=sys.argv[2]
##            self.num_samples = int(1000)
            self.num_samples =int(sys.argv[3])
            self.read_data_test(file_name_test)
            self.query_variable=self.rand_vars[-1]
            for instance in range(0,len(self.rv_all_values)):
                 prob_query,self.query_evidence =self.creat_query(self.rv_all_values[instance])
                 self.prob_dist = self.weighting_sampling()
                 self.predictions[instance]=self.prob_dist
                 #print("for instance "+str(self.rv_all_values.index(instance))+" probability_distribution="+str(self.prob_dist))
            #self.calculate_scoring_functions()
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
        
        for value in self.rv_key_values[self.query_variable]:
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
    
    def get_gaussian_probability(self, x, mean, stdev):
        e_val = -0.5*np.power((float(x)-mean)/stdev, 2)
        probability = (1/(stdev*np.sqrt(2*np.pi))) * np.exp(e_val)
        return probability

    def change_w(self,var,X):
        var_value=self.query_evidence[var]
        w_prob={}
        parent_string_value=""
        parent_sample={}
        parents=bnu.get_parents(var, self.bn)
        if parents is None:
            pdf="PDF("+var+")"
            mean_var=self.bn[pdf][var_value][0]
            stdev_var=self.bn[pdf][var_value][1]
            p=self.get_gaussian_probability(X, mean_var,stdev_var)
        elif len(parents.split(","))==1:
            pdf="PDF("+var+"|"+parents+")"
            maen_withparent=self.bn[pdf][0]
            mean_var=float(maen_withparent.split("*")[0])*float(X[parents])+float(maen_withparent.split("+")[1])
            stdev_var=float(self.bn[pdf][1])
            p=self.get_gaussian_probability(var_value,mean_var,stdev_var)
        return p
    
    def get_sampled_value(self, V, sampled):
        # get the conditional probability distribution (cpt) of variable V
        parents = bnu.get_parents(V, self.bn)
        pdf1 = {}
        prob_mass = 0
        # generate a cumulative distribution for random variable V
        if parents is None:
            for value in self.rv_key_values[self.predictor_variable]:
                pdf="PDF("+V+")"
                mean_var=float(self.bn[pdf][0])
                stdev_var=float(self.bn[pdf][1])
                probability=self.get_gaussian_probability(value,mean_var,stdev_var)
                prob_mass += probability
                pdf1[value] = prob_mass

        # check that the cpt sums to 1 (or almost)
        if prob_mass < 0.999 and prob_mass > 1:
            print("ERROR: CPT=%s does not sum to 1" % (cpt))
            exit(0)

        return self.sampling_from_cumulative_distribution(pdf1,mean_var,stdev_var)

    def sampling_from_cumulative_distribution(self, cumulative,mean,stdev):
        random_number = np.random.normal(mean,stdev,1)
        p=self.get_gaussian_probability(random_number,mean,stdev)
        for value, probability in cumulative.items():
            if p <= probability:
                random_number = np.random.normal(mean,stdev,1)
                p=self.get_gaussian_probability(random_number,mean,stdev)
                return value.split("|")[0]

        print("ERROR couldn't do sampling from:")
        print("cumulative_dist="+str(cumulative))
        exit(0)

    def calculate_log_lilelihood(self):
        LL = 0

        # iterate over all instances in the training data
        for instance in self.rv_all_values:
            predictor_value = instance[len(instance)-1]

            # iterate over all random variables except the predictor var.
            for value_index in range(0, len(instance)):
                variable = self.rand_vars[value_index]
                value = instance[value_index]
                mean = self.gaussian_means[variable][predictor_value]
                stdev = self.gaussian_stdevs[variable][predictor_value]
                prob = self.get_gaussian_probability(value, mean, stdev)

                LL += math.log(prob)

            # accumulate the log prob of the predictor variable
            
            if self.verbose is True:
                print("LL: %s -> %f" % (instance, LL))
        return LL

    def calculate_bayesian_information_criterion(self, LL):
        penalty = 0
        num_params = 2  # mean and stdev

        for variable in self.rand_vars:
            local_penalty = (math.log(self.num_data_instances)*num_params)/2
            penalty += local_penalty

        BIC = LL - penalty
        return BIC
    
    
    def calculate_scoring_functions(self):
        print("\nCALCULATING LL and BIC on training data...")
        LL = self.calculate_log_lilelihood()
        BIC = self.calculate_bayesian_information_criterion(LL)
        print("LL score="+str(LL))
        print("BIC score="+str(BIC))
    
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
