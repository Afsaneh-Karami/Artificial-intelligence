
class NB_Classifier:
    #learning_param=['stroke','age|stroke','smoking_status|stroke','gender|avg_glucose_level,smoking_status','avg_glucose_level|smoking_status,hypertension','hypertension','work_type','heart_disease','ever_married|heart_disease','bmi|ever_married,smoking_status,work_type']
    learning_param=['target','age|target', 'sex|target', 'cp|target', 'trestbps|target', 'chol|target', 'fbs|target', 'restecg|target', 'thalach|target', 'exang|target', 'oldpeak|target', 'slope|target', 'ca|target', 'thal|target']
    rand_vars = []
    rv_key_values = {}
    rv_all_values = []
    predictor_variable = None
    num_data_instances = 0
    probabilities = {}
    predictions = []

    def __init__(self, file_name, fitted_model=None):
        self.read_data(file_name)
        

    def read_data(self, data_file):
        print("\nREADING data file %s..." % (data_file))
        print("---------------------------------------")

        self.rand_vars = []
        self.rv_key_values = {}
        self.rv_all_values = []

        with open(data_file) as csv_file:
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

##        self.predictor_variable = self.rand_vars[len(self.rand_vars)-1]

        print("RANDOM VARIABLES=%s" % (self.rand_vars))
        print("VARIABLE KEY VALUES=%s" % (self.rv_key_values))
        print("|data instances|=%d" % (self.num_data_instances))
        #print("Learning_parameter=%s" %(self.probability))
        count_dict=self.probability()
        self.create_cpt(count_dict)

    def update_variable_key_values(self, values):
        for i in range(0, len(self.rand_vars)):
            variable = self.rand_vars[i]
            key_values = self.rv_key_values[variable]
            value_in_focus = values[i]
            if value_in_focus not in key_values:
                self.rv_key_values[variable].append(value_in_focus)
                
    def probability(self):
        count_dict={}
        for i in self.learning_param:
            query_item_prob={}
            count_value=0
            query=i
            query_multi=query.split("|")
            if len(query_multi)==1:
                pos=self.rand_vars.index(query)
                for k in self.rv_key_values[query]:
                    for j in range(0,len(self.rv_all_values)):
                        if self.rv_all_values[j][pos]==k:
                            count_value=count_value+1
                    prob=count_value/len(self.rv_all_values)
                    query_item_prob[k]=prob
                    count_value=0
                    count_dict["CPT("+query+")"]=query_item_prob          
            else:
                query_val=[]
                query_comb=[]
                position=[]
                diff_comb={}
                parent_child=query.split("|")
                child=parent_child[0]
                parents=parent_child[1].split(",")
                pos_child=self.rand_vars.index(child)
                position.append(pos_child)
                for parent in parents:
                    pos_parent=self.rand_vars.index(parent)
                    position.append(pos_parent)
                query_val.append(self.rv_key_values[child])
                for parent in parents:
                    query_val.append(self.rv_key_values[parent])
                co=0
                
                if len(parents)==1:
                    
                    for child_p in range(0,len(query_val[0])):
                        for parent_1 in range(0,len(query_val[1])):
                            query_comb.append([query_val[0][child_p],query_val[1][parent_1]])
                            
                elif len(parents)==2:
                    for child_p in range(0,len(query_val[0])):
                        for parent_1 in range(0,len(query_val[1])):
                            for parent_2 in range(0,len(query_val[2])):
                                query_comb.append([query_val[0][child_p],query_val[1][parent_1],query_val[2][parent_2]])
                                co=co+1
                elif len(parents)==3:
                    for child_p in range(0,len(query_val[0])):
                        for parent_1 in range(0,len(query_val[1])):
                            for parent_2 in range(0,len(query_val[2])):
                                for parent_3 in range(0,len(query_val[3])):
                                    query_comb.append([query_val[0][child_p],query_val[1][parent_1],query_val[2][parent_2],query_val[3][parent_3]])
                                    co=co+1
                                    
                for con in query_comb:
                    count_Parent_value=0
                    count_child_value=0
                    for data_value in (self.rv_all_values):
                        if len(con)==2:
                            cpt_key=con[0]+"|"+con[1]
                            pose=int(position[1])
                            if data_value[pose]== con[1]:
                                count_Parent_value=count_Parent_value+1
                                if data_value[position[0]]==con[0]:
                                    count_child_value=count_child_value+1
                            else:
                                continue
                        elif len(con)==3:
                            cpt_key=con[0]+"|"+con[1]+","+con[2]
                            pose1=int(position[1])
                            pose2=int(position[2])
                            if data_value[pose1]== con[1] and data_value[pose2]== con[2]:
                                count_Parent_value=count_Parent_value+1
                                if data_value[position[0]]==con[0]:
                                    count_child_value=count_child_value+1
                            else:
                                continue
                        elif len(con)==4:
                            cpt_key=con[0]+"|"+con[1]+","+con[2]+","+con[3]
                            pose1=int(position[1])
                            pose2=int(position[2])
                            pose3=int(position[3])
                            if data_value[pose1]== con[1] and data_value[pose2]== con[2] and data_value[pose3]== con[3]:
                                count_Parent_value=count_Parent_value+1
                                if data_value[position[0]]==con[0]:
                                    count_child_value=count_child_value+1
                            else:
                                continue
                    
                    prob=(count_child_value+1)/(count_Parent_value+len(self.rv_key_values[child]))
                    diff_comb[cpt_key]=prob
                        
                count_dict["CPT("+query+")"]=diff_comb
        return count_dict
    
    def create_cpt(self,count_dict):
        for key,value in count_dict.items():
            print(key+":")
            for key_1,value_1 in value.items():
                print(key_1+"="+str(value_1)+";")
            print()
    
file_name_train="heart-data-discretized-train.csv"
NB_Classifier(file_name_train)

