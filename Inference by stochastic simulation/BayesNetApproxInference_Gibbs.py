
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
            print("USAGE> BayesNetApproxInference_Gibbs.py [your_config_file.txt] [query] [num_samples]")
            print("EXAMPLE> BayesNetApproxInference_Gibbs.py \ config-heart-NaiveBayes.txt \ 10000")
        else:
            file_name = sys.argv[1]
            prob_query =sys.argv[2]
            self.num_samples = int(sys.argv[3])
##            file_name = "config-heart-NaiveBayes.txt"
##            prob_query ="P(target|sex=0,cp=3)"
##            self.num_samples = int(2000)
            super().__init__(file_name)
            self.query = bnu.tokenise_query(prob_query)
            self.prob_dist = self.Gibbs_sampling()
            print("probability_distribution="+str(self.prob_dist))

    def Gibbs_sampling(self):
        print("\nSTARTING rejection sampling...")
        query_variable = self.query["query_var"]
        evidence = self.query["evidence"]
        C = {}
        nonevidence_variables=self.del_evidence_sampling()
        # initialise vector of counts
        for value in self.bn["rv_key_values"][query_variable]:
            value = value.split("|")[0]
            C[value] = 0

        # loop to increase counts when the sampled vector consistent w/evidence
        for i in range(0, self.num_samples):
            if i==0:
                X = self.prior_sample(nonevidence_variables)
            else:
                rand_choice=random.randint(-1,len(nonevidence_variables)-1)
                rand_node=nonevidence_variables[rand_choice]
                p=self.Markov_blanket_probability(rand_node,X)
                rand_node_value=self.sampling_from_markov_blanket(rand_node,p)
                X[rand_node]=rand_node_value
            Y=X[query_variable]
            C[Y] += 1 

        return bnu.normalise(C)
    
    def del_evidence_sampling(self):
        var_evidence=[]
        random_variables=self.bn["random_variables"]
        for key,var in self.query["evidence"].items():
            var_evidence.append(key)
        for var in var_evidence:
            if var in random_variables:
                random_variables.remove(var)
        return random_variables
    
    def prior_sample(self,var):
        X = {}
        sampled_var_values = {}
        for variable in var:
            X[variable] = self.get_sampled_value(variable, sampled_var_values)
            sampled_var_values[variable] = X[variable]   
        return X
            
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
    
    def Markov_blanket_probability(self,node,X):
        p={}
        parent_value_X=""
        parent_X=""
        parent_of_child_X_value=""
        parent_of_child_y=""
        effective_node=self.Markov_blanket_mem(node)
        
        if "parent" not in effective_node:   
            cpt="CPT("+node+")"
            p=self.bn[cpt]
            
        else:
            for parent in effective_node["parent"]:
                parent_node=X[parent]
                parent_X=parent_X+parent
                parent_value_X=parent_value_X+parent_node
            cpt="CPT("+node+"|"+parent_X+")"
            
            for value in bnu.get_domain_values(node, self.bn):
                y=value+"|"+parent_value_X
                prob=self.bn[cpt][y]
                p[y]=prob
        if "parent_of_child" not in effective_node:
            pp=1
        else:
            for child in effective_node["child"]:
                if bnu.get_parents(str(child), self.bn)==node:
                        continue
                else:
                    parents_of_child=bnu.get_parents(str(child), self.bn)
                    parent_of_child=parents_of_child.split(",")
                    for value in parent_of_child:
                        parent_of_child_X=X[value]
                        parent_of_child_X_value=parent_of_child_X_value+parent_of_child_X
                    parent_of_child_y=child+"|"+parent_of_child_X_value
                    cpt="CPT("+node+"|"+effective_node["child"]+")"
                    pp=self.bn[cpt][parent_of_child_y]
                    for key,value in p.items():
                        p[key]=p[key]*pp         
        return p
        
    def Markov_blanket_mem(self,var):
        Markov_blanket_items=[]
        Markov_blanket_list={}
        parent=bnu.get_parents(str(var), self.bn)
        if parent is not None:
            parent=parent.split(",")
            Markov_blanket_list["parent"]=parent
            
        if len(self.get_child(str(var)))>=1:
            Markov_blanket_list["child"]=self.get_child(str(var))
            for child in Markov_blanket_list["child"]:
                parent_of_child=bnu.get_parents(str(child), self.bn)
                for items in parent_of_child:
                    if parent is not None:
                        if items in Markov_blanket_list["parent"]:
                            continue
                    else:
                        Markov_blanket_items.append(items)
            Markov_blanket_list["parent_of_child"]=Markov_blanket_items  
        return Markov_blanket_list
        
    def get_child(self,var):
        child=[]
        for rand_var in self.bn["random_variables"]:
            if ("P("+rand_var+"|"+ var+")") in self.bn["structure"]:
                child.append(rand_var)
        return child
    
    def sampling_from_markov_blanket(self, node,p):
        k=0
        random_number = random.random()
        for key,value in p.items():
            k=k+value
            p[key]=k
        for key,value in p.items():
            if random_number <= value:
                p_node=key.split("|")[0]    
        return p_node
           
        print("ERROR couldn't do sampling from:")
        exit(0)

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

