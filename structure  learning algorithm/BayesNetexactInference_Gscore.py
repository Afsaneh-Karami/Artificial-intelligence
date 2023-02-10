
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
        if len(sys.argv) != 2:
            print("USAGE> BayesNetApproxInference.py [your_config_file.txt] [your_test_file.csv]")
            print("EXAMPLE> BayesNetApproxInference.py config-heart-NaiveBayes.txt heart-data-discretized-test.csv")
        else:
            
            self.predictions={}
##            file_name = "config-stroke-NaiveBayes.txt"
            file_name =sys.argv[1]
            super().__init__(file_name)
            self.inference_time = time.time()
##            file_name_test="stroke-data-discretized-test.csv"
            file_name_test=sys.argv[2]
            self.read_data_test(file_name_test)
            self.calculate_scoring_functions()
            self.inference_time = time.time() - self.inference_time
            
            
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

    
    def calculate_scoring_functions(self):
        print("\nCALCULATING LL and BIC on training data...")
        LL,num = self.calculate_log_lilelihood()
        BIC = self.calculate_bayesian_information_criterion(LL,num)
        print("LL score="+str(LL))
        print("BIC score="+str(BIC))

    def calculate_log_lilelihood(self):
        LL = 0
        count=-1
        # iterate over all instances in the training data
        for instance in self.rv_all_values:
            count=count+1
            prob_dist={}
            num={}
            #predictor_value = instance[len(instance)-1]

            # iterate over all random variables except the predictor var.
            for value_index in range(0, len(instance)):
                variable = self.rand_vars[value_index]
                value = instance[value_index]
                predictor_value =self.find_parent(variable)
                prob,numebr_prob= self.prob(variable,predictor_value,count)
                LL += math.log(prob)
                if variable in num:
                    continue
                else:
                    num[variable]=numebr_prob
        return LL,num
    
    def prob(self,child,parent,count):
        parent_position=[]
        parent_value=[]
        child_pose = self.rand_vars.index(child)
        child_value=self.rv_all_values[count][child_pose]
        if parent==None:
            cpt="CPT("+child+")"
            cpt_item=child_value
            probability=self.bn[cpt][cpt_item]
            numebr_prob=len(self.bn[cpt])
        else:
            cpt_parent=""
            cpt="CPT("+child+"|"+parent+")"
            for parent_item in parent.split(","):
                parent_pose=self.rand_vars.index(parent_item)
                parent_position.append(parent_pose)
                parent_item_value=self.rv_all_values[count][parent_pose]
                parent_value.append(parent_item_value)
            for item in range(0,len(parent_value)):
                cpt_parent=cpt_parent+parent_value[item]+","
            cpt_item=child_value+"|"+cpt_parent[0:len(cpt_parent)-1]
            probability=self.bn[cpt][cpt_item]
            numebr_prob=len(self.bn[cpt])
        return probability,numebr_prob
    

    def find_parent(self,variable):
        count=0
        for structure_item in self.bn["structure"]:
            element=structure_item[2:len(structure_item)-1]
            if element.split("|")[0]==variable:
                if len(element.split("|"))>1:
                    parent=element.split("|")[1]
                    count=1
                else:
                    parent=None
                    count=1
            else:
                continue
        if count==0:
            parent=None
        return parent            
                
            
        
    def calculate_bayesian_information_criterion(self, LL,num):
        penalty = 0
        for variable in self.rand_vars:
            num_params = num[variable]
            local_penalty = (math.log(self.num_data_instances)*num_params)/2
            penalty += local_penalty

        BIC = LL - penalty
        return BIC
    
##   
        
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
