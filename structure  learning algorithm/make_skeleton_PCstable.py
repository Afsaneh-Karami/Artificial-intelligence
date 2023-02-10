
import sys
from causallearn.utils.cit import CIT


class ConditionalIndependence:
    chisq_obj = None
    rand_vars = []
    rv_all_values = []
    parents=[]
    links=[]

    def __init__(self, file_name):
        data = self.read_data(file_name)
        self.chisq_obj = CIT(data, "chisq")
        

    def read_data(self, data_file):
        
        with open(data_file) as csv_file:
            for line in csv_file:
                line = line.strip()
                if len(self.rand_vars) == 0:
                    self.rand_vars = line.split(',')
                else:
                    values = line.split(',')
                    self.rv_all_values.append(values)        
        return self.rv_all_values  

    def get_var_index(self, target_variable):
        for i in range(0, len(self.rand_vars)):
            if self.rand_vars[i] == target_variable:
                return i
        print("ERROR: Couldn't find index of variable "+str(target_variable))
        return None

    def get_var_indexes(self, parent_variables):
        if len(parent_variables) == 0:
            return None
        else:
            index_vector = []
            for parent in parent_variables:
                index_vector.append(self.get_var_index(parent))
            return index_vector

    def compute_pvalue(self, variable_i, variable_j, parents_i):
        var_i = self.get_var_index(variable_i)
        var_j = self.get_var_index(variable_j)
        par_i = None
        p = self.chisq_obj(var_i, var_j, par_i)
        return p
    
    def make_link(self):
        
        for level in range(0,len(self.rand_vars)-2):
            if level==0:
                parents=None
                Vi =self.rand_vars[-1]
                for node_link in self.rand_vars:
                    if (node_link==Vi):
                        continue
                    else:
                        Vj=node_link
                        p=self.compute_pvalue(Vi, Vj, parents)
                        if p<=0.01:
                            self.links.append((Vi,Vj))
                            
                for node in self.rand_vars:
                    Vi=node
                    for node_link in self.rand_vars:
                        if (node==node_link) or ([node_link,node] in self.links) or node==self.rand_vars[-1]:
                            continue
                        else :
                            Vj=node_link
                            p=self.compute_pvalue(Vi, Vj, parents)
                            if p<=0.01:
                                self.links.append((Vi,Vj))
            elif level!=0:
                parent=[]
                for link in self.links:
                    number_parent=level
                    for index in range(0,number_parent):
                        for node in self.rand_vars:
                            if (node in parent) or node==link[1] or node==link[0]:
                                continue
                            if ((link[0],node) in self.links) and ((link[1],node) in self.links) and ((node,link[0]) in self.links) and ((node,link[1]) in self.links):
                                parent.append(node)
                            if len(parent)==number_parent:
                                p=self.compute_pvalue(Vi, Vj, parent)
                                parent.clear()
                                if p>=0.01:
                                    if link in self.links:
                                        self.links.remove(link)
                                       
        print(len(self.links))
        return self.links
   
        
data_file ="stroke-data-discretized-train.csv"
ci = ConditionalIndependence(data_file)
m=ci.read_data(data_file)
y=ci.make_link()
print(y)
