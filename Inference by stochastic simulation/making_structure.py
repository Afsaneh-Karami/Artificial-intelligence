G2=[('stroke', 'age'), ('stroke', 'hypertension'), ('stroke', 'heart_disease'), ('stroke', 'ever_married'), ('stroke', 'work_type'), ('stroke', 'avg_glucose_level'), ('stroke', 'bmi'), ('stroke', 'smoking_status'), ('gender', 'age'), ('gender', 'heart_disease'), ('gender', 'work_type'), ('gender', 'avg_glucose_level'), ('gender', 'bmi'), ('gender', 'smoking_status'), ('age', 'gender'), ('age', 'hypertension'), ('age', 'heart_disease'), ('age', 'ever_married'), ('age', 'work_type'), ('age', 'avg_glucose_level'), ('age', 'bmi'), ('age', 'smoking_status'), ('age', 'stroke'), ('hypertension', 'age'), ('hypertension', 'heart_disease'), ('hypertension', 'ever_married'), ('hypertension', 'work_type'), ('hypertension', 'avg_glucose_level'), ('hypertension', 'bmi'), ('hypertension', 'smoking_status'), ('hypertension', 'stroke'), ('heart_disease', 'gender'), ('heart_disease', 'age'), ('heart_disease', 'hypertension'), ('heart_disease', 'ever_married'), ('heart_disease', 'work_type'), ('heart_disease', 'avg_glucose_level'), ('heart_disease', 'bmi'), ('heart_disease', 'smoking_status'), ('heart_disease', 'stroke'), ('ever_married', 'age'), ('ever_married', 'hypertension'), ('ever_married', 'heart_disease'), ('ever_married', 'work_type'), ('ever_married', 'avg_glucose_level'), ('ever_married', 'bmi'), ('ever_married', 'smoking_status'), ('ever_married', 'stroke'), ('work_type', 'gender'), ('work_type', 'age'), ('work_type', 'hypertension'), ('work_type', 'heart_disease'), ('work_type', 'ever_married'), ('work_type', 'avg_glucose_level'), ('work_type', 'bmi'), ('work_type', 'smoking_status'), ('work_type', 'stroke'), ('avg_glucose_level', 'gender'), ('avg_glucose_level', 'age'), ('avg_glucose_level', 'hypertension'), ('avg_glucose_level', 'heart_disease'), ('avg_glucose_level', 'ever_married'), ('avg_glucose_level', 'work_type'), ('avg_glucose_level', 'bmi'), ('avg_glucose_level', 'smoking_status'), ('avg_glucose_level', 'stroke'), ('bmi', 'gender'), ('bmi', 'age'), ('bmi', 'hypertension'), ('bmi', 'heart_disease'), ('bmi', 'ever_married'), ('bmi', 'work_type'), ('bmi', 'avg_glucose_level'), ('bmi', 'smoking_status'), ('bmi', 'stroke'), ('smoking_status', 'gender'), ('smoking_status', 'age'), ('smoking_status', 'hypertension'), ('smoking_status', 'heart_disease'), ('smoking_status', 'ever_married'), ('smoking_status', 'work_type'), ('smoking_status', 'avg_glucose_level'), ('smoking_status', 'bmi'), ('smoking_status', 'stroke')]
structure={}
y=""
x=""
child_list=[]
parent_list=[]
for item in G2:
    parent_v=""
    child=item[1]
    if child not in child_list:
        child_list.append(child)
    for i in G2:
        if i[1]==child:
            parent=i[0]
            parent_v=parent_v+parent+","
    parent_modify=parent_v[:len(parent_v)-1]
    structure[child]=parent_modify
for item in G2:
    if item[0] not in child_list:
        if item[0] not in parent_list:
            parent_list.append(item[0])
for parent in parent_list:
    x=x+"P("+parent+")"+";"
x_modify=x[:len(x)-1]   
for key,value in structure.items():
    y=y+"P("+key+"|"+value+")"+";"
y_modify=y[:len(y)-1]
print(x_modify+";"+y_modify)

