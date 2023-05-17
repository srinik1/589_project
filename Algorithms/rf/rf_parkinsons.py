import numpy as np
import math
import pandas as pd
import random
from sklearn.utils import shuffle
from collections import Counter
import sklearn.metrics as ms
import matplotlib.pyplot as plt

def entropy_of_dataset(dataset, column="class") :
    row_count = len(dataset)
    value_counts = dataset[column].value_counts()

    if(row_count == 0) :
        # return 0
        print("Entropy For 0 count CALLED!!!")
        print(dataset)    
        print("Row count :", row_count)
        print("Value counts", value_counts)
    return -((value_counts/row_count)*(np.log(value_counts/row_count)/np.log(2))).sum()

def gini_of_dataset(dataset, column="class") :
    row_count = len(dataset)
    value_counts = dataset[column].value_counts()
    if(row_count == 0) :
        # return 0
        print("Gini For 0 count CALLED!!!")
        print(dataset)    
        print("Row count :", row_count)
        print("Value counts", value_counts)
    return 1-((value_counts/row_count)**2).sum()

def getAttributeType(attrib, str_attrib_list) :
    if attrib in str_attrib_list :
        return "string"
    return "integer"

def getNextAttributeCondition(attribList, dataview, useGini, str_attrib_list) :
    if(len(attribList)==1) :
        return attribList[0]
    # lowest entropy will give maximum info gain
    # print("GetNextAttribute Here")
    lowest_attrib = None
    lowest_thresh = None
    lowest_entropy = math.inf

    if(len(attribList)==0) :
        print("WARNINGNJJNN")
        return None
    for attrib in attribList :
        #For each attrib, calculate the new entopy for each view and avg final entropy. 
        avg_entropy = 0

        attr_type = getAttributeType(attrib, str_attrib_list)
        
        if attr_type != "string" :
            threshHoldVal = dataview[attrib].mean()
            data_val = dataview[dataview[attrib] <= threshHoldVal]
            if(len(data_val)!=0) :
                if(useGini) :
                    val_entr = gini_of_dataset(data_val)
                else :
                    val_entr = entropy_of_dataset(data_val)
                avg_entropy += (len(data_val)/len(dataview))*val_entr

            data_val = dataview[dataview[attrib] > threshHoldVal]
            if(len(data_val)!=0) :
                if(useGini) :
                    val_entr = gini_of_dataset(data_val)
                else :
                    val_entr = entropy_of_dataset(data_val)
                avg_entropy += (len(data_val)/len(dataview))*val_entr

            #print("Entropy, attrib", avg_entropy, attrib)
            if(avg_entropy < lowest_entropy) :
                lowest_attrib = attrib
                lowest_thresh = threshHoldVal
                lowest_entropy = avg_entropy
        else :
            #take unique values of task, and get view based on values on them and calculate the entropy
            for each_val in dataview[attrib].unique():
                data_val = dataview[dataview[attrib] == each_val]
                if(len(data_val)!=0) :
                    if(useGini) :
                        val_entr = gini_of_dataset(data_val)
                    else :
                        val_entr = entropy_of_dataset(data_val)
                    avg_entropy += (len(data_val)/len(dataview))*val_entr

            #print("Entropy, attrib", avg_entropy, attrib)
            if(avg_entropy < lowest_entropy) :
                lowest_attrib = attrib
                lowest_thresh = None
                lowest_entropy = avg_entropy
    
    if(lowest_attrib == None) :
        print("WARNINGNJJNNWARNINGNJJNNWARNINGNJJNNWARNINGNJJNNWARNINGNJJNN")

    return lowest_attrib, lowest_thresh

def limit_depth_percent(list, maxDepth) :
    item_counter = Counter()
    for item in list :
        item_counter.update(str(item))

    most_common = item_counter.most_common(1)[0]
    percentage = most_common[1] / len(list)
    
    if (percentage >= (1-maxDepth)) :
        return True
    return False


def most_frequent(list, output_type) :
    item_counter = Counter()
    for item in list :
        item_counter.update(str(item))

    frequent_items = item_counter.most_common()
    # If there is only one item or if the most common item is not tied with any others, return it
    if len(frequent_items) == 1 or frequent_items[0][1] != frequent_items[1][1]:
        most_common = frequent_items[0][0]
    else:
        # print("Random")
        # If the most common item is tied with one or more others, randomly select one of them
        tied_items = [item for item in frequent_items if item[1] == frequent_items[0][1]]
        most_common = random.choice(tied_items)[0]
    
    if output_type == "str" :
        return most_common
    return int(most_common)

def getAttribListSquareRoot(list) :
    num_elements = math.floor(math.sqrt(len(list)))
    # Get a random subset of the list
    return random.sample(list, num_elements)


def recursive_tree_builder(dataset, useGini, maxDepth, output_type, str_attrib_list) :
    attribListTobeTested = getAttribListSquareRoot(list(dataset)[:-1])

    # print(dataset)
    newNode = Node()

    ##Stopping Criteria
    #If All Instances in dataset belong to same class y
    target_class_list = dataset['class'].values.tolist()
    majority_decision_value = most_frequent(target_class_list, output_type)

    if(len(set(target_class_list)) == 1) :
        newNode.isLeafNode= True
        newNode.decisionValue = target_class_list[0]
        return newNode
    
    if(maxDepth > 0 and maxDepth < 1) :
        if(limit_depth_percent(target_class_list, maxDepth)) :
            newNode.isLeafNode =  True
            newNode.decisionValue = majority_decision_value
            return newNode

    #Get best attribute to split based on Dataset
    (bestAttribute, threshHold) = getNextAttributeCondition(attribListTobeTested,dataset, useGini, str_attrib_list)
    #print(bestAttribute)
    newNode.checkAttribute = bestAttribute
    newNode.isDecisionNode = True
    newNode.valueToCheck = threshHold
    newNode.majorityValue = majority_decision_value

    #CreatesSubtrees /trees
    if threshHold != None :
        data_containing_val = dataset[dataset[bestAttribute] <= threshHold]
        if(len(data_containing_val)==0) :
            tree_v = Node()
            tree_v.isLeafNode = True
            tree_v.decisionValue = majority_decision_value
        else :
            tree_v = recursive_tree_builder(data_containing_val, useGini, maxDepth, output_type, str_attrib_list)
        newNode.children.append(tree_v)
        
            
        data_containing_val = dataset[dataset[bestAttribute] > threshHold]
        if(len(data_containing_val)==0) :
            tree_v = Node()
            tree_v.isLeafNode = True
            tree_v.decisionValue = majority_decision_value
        else :
            tree_v = recursive_tree_builder(data_containing_val, useGini, maxDepth, output_type, str_attrib_list)
        newNode.children.append(tree_v)
        
    else :
        uniqueAttribValues = dataset[bestAttribute].unique()
        for each_val in uniqueAttribValues:
            data_containing_val = dataset[dataset[bestAttribute] == each_val]
            if(len(data_containing_val)==0) :
                print("ERRORO SHOULDNOT BE CALLED")
                exit()
                tree_v = Node()
                tree_v.isLeafNode = True
                tree_v.decisionValue = majority_decision_value
            else :
                tree_v = recursive_tree_builder(data_containing_val, useGini, maxDepth, output_type, str_attrib_list)
            newNode.children.append(tree_v)
            newNode.conditionValueList.append(each_val)
        if(len(uniqueAttribValues) ==0) :
            print("Test called")
            tree_v = Node()
            tree_v.isLeafNode = True
            tree_v.decisionValue = majority_decision_value
            return tree_v
            
    return newNode

def traverse_and_predict(tree_node, row) :
    if(tree_node.isLeafNode) :
        return tree_node.decisionValue
    elif (tree_node.isDecisionNode) :
        if(tree_node.valueToCheck != None) :
            if(row[tree_node.checkAttribute]<= tree_node.valueToCheck) :
                return traverse_and_predict(tree_node.children[0], row)
            if(row[tree_node.checkAttribute]>tree_node.valueToCheck) :
                return traverse_and_predict(tree_node.children[1], row)
        else :
            try:
                idx = tree_node.conditionValueList.index(row[tree_node.checkAttribute])
            except ValueError as ve:
                return tree_node.majorityValue
            return traverse_and_predict(tree_node.children[idx], row)

def get_accuracy(rootnode, testdata) :
    correct_predictions = 0
    wrong_predictions = 0
    for index, row in testdata.iterrows():
        prediction = traverse_and_predict(rootnode, row)
        if(prediction==row['class']) :
            correct_predictions += 1
        else :
            wrong_predictions += 1
    return correct_predictions/len(testdata)

def get_ensemble_accuracy(rootnodes, testdata, output_type) :
    predictions = []
    actual = []
    correct_predictions = 0

    for index, row in testdata.iterrows():
        predictionsOfEnsemble = []
        for root in rootnodes :
            predictionsOfEnsemble.append(traverse_and_predict(root, row))
        majority = most_frequent(predictionsOfEnsemble, output_type)
        
        if(output_type != "str") :
            predictions.append(int(majority))
            actual.append(int(row['class']))
        else :
            predictions.append(majority)
            actual.append(row['class'])

        #print(majority,str(row['class']))
        if(majority==row['class'] or majority==str(row['class'])) :
            correct_predictions += 1
    accuracies = ms.accuracy_score(actual,predictions)
    precisions = np.mean(ms.precision_score(actual, predictions, average=None))
    recalls = np.mean(ms.recall_score(actual,predictions, average=None))
    f1Score = np.mean(ms.f1_score(actual, predictions, average=None))

    return (accuracies, precisions, recalls, f1Score)

def bagging(df, nTree) :
    bags = []
    df = shuffle(df)
    for i in range(nTree) :
        indices = np.random.choice(range(len(df)), size=len(df), replace=True)
        bag = df.iloc[indices]
        bags.append(bag)
    return bags

class Node:
    def __init__(self) -> None:
        self.checkAttribute = None
        self.isLeafNode = False
        self.decisionValue = None
        self.isDecisionNode = False
        self.valueToCheck = None
        self.majorityValue = None

        self.left =  None
        self.middle = None
        self.right = None

        self.children = []
        self.conditionValueList = []
    
    def __str__(self, level=0):
        if(self.isDecisionNode) :
            val_to_print = self.checkAttribute + " " + str(self.valueToCheck)
        elif (self.isLeafNode) :
            val_to_print = "Leaf node: "+ str(self.decisionValue)
        ret = "\t"*level+repr(val_to_print)+"\n"
        
        if(self.left) :
            ret += self.left.__str__(level+1)
        if(self.middle) :
            ret += self.middle.__str__(level+1)
        if(self.right) :
            ret += self.right.__str__(level+1)
        
        return ret
    
    def __repr__(self):
        return '<tree node representation>'

def main() :
    #HardCode all variable names for now - Take arguments later
    file_name = "parkinsons.csv"
    input_dataset = pd.read_csv(file_name)
    # file_name = "datasets/hw3_house_votes_84.csv"
    # input_dataset = pd.read_csv(file_name)#, sep="\t")

    output_type = "integer"
    # str_attrib_list = ["Gender","Married", "Dependents", "Education", "Self_Employed", "Credit_History","Property_Area"]
    str_attrib_list = []
    # input_dataset = input_dataset.drop("Loan_ID", axis=1)

    k_fold_num = 10

    maxDepth = 1 #Default 1 => Change to 0.85
    useGini = False

    

    # print(input_dataset)
    total_count = len(input_dataset)
    value_count = {}
    val_df = {}
    each_fold_count = total_count//k_fold_num

    each_fold_val_count = {}

    for val in input_dataset['class'].unique() :
        value_count[val] = (input_dataset['class'] == val).sum()
        val_df[val] = input_dataset[input_dataset['class'] == val]
        each_fold_val_count[val] = math.floor(each_fold_count * (value_count[val]/total_count))
    
    k_fold_df = []

    for i in range(k_fold_num) :
        fold = pd.DataFrame(columns=input_dataset.columns)
        for val in value_count:
            sample = val_df[val].sample(n = each_fold_val_count[val])
            val_df[val] = val_df[val].drop(sample.index)
            fold = pd.concat([fold, sample])
        k_fold_df.append(fold)
    
    for val in value_count:
        k_fold_df[k_fold_num-1] = pd.concat([k_fold_df[k_fold_num-1], val_df[val]])

    k_fold_total_data = {}
    for i in range(k_fold_num) :

        test_df_set = k_fold_df[i]

        train_df_set = pd.DataFrame(columns=input_dataset.columns)
        for j in range(k_fold_num) :
            if j != i :
                train_df_set = pd.concat([train_df_set, k_fold_df[j]])
        
        k_fold_total_data[i] = {"test" : test_df_set, "train": train_df_set}

    accuracies = []
    recalls = []
    precisions = []
    f1_scores = []


    # nTreeList = [1,5,10,20,30,40,50]
    nTreeList = [10,20]
    for nTree in nTreeList :
        bags = {}
        #Bagging Start
        print("nTree",nTree)
        for i in range(k_fold_num) :
            bags[i] = bagging(k_fold_total_data[i]['train'], nTree)
        
        randomForest = {}
        for i in range(k_fold_num) : #k_fold_num
            randomForest[i] = []
            for df_train in bags[i] :
                dtRootNode =  recursive_tree_builder(df_train, useGini, maxDepth, output_type, str_attrib_list)
                # print(dtRootNode)
                randomForest[i].append(dtRootNode)

        
        avg_accuracy = 0
        avg_precision = 0
        avg_recall =0
        avg_f1score = 0
        

        for i in range(k_fold_num) : #K_fold_num
            (accuracies_a, precisions_a, recalls_a, f1Score_a) = get_ensemble_accuracy(randomForest[i], k_fold_total_data[i]['test'], output_type)
            # print(accuracies)
            avg_accuracy += accuracies_a
            avg_precision += precisions_a
            avg_recall += recalls_a
            avg_f1score += f1Score_a
        accuracies.append(avg_accuracy/k_fold_num)
        precisions.append(avg_precision/k_fold_num)
        recalls.append(avg_recall/k_fold_num)
        f1_scores.append(avg_f1score/k_fold_num)
    
    fig, ax = plt.subplots()
    print("Accuracies : ", accuracies)
    print("Recalls : ", recalls)
    print("Precisions : ", precisions)
    print("F1 scores : ", f1_scores)

    ax.plot(nTreeList, accuracies, '-o', label='Accuracies')
    ax.plot(nTreeList, recalls, '-o', label='Recalls')
    ax.plot(nTreeList, precisions, '-o', label='Precisions')
    ax.plot(nTreeList, f1_scores, '-o', label='F1 Scores')

    # Add labels and legend
    ax.set_xlabel('nTree values')
    ax.set_ylabel('metrics Range')
    ax.set_title('nTree Metrics Graph for Cancer Dataset')
    ax.legend()

    plt.show()

    

if __name__ == '__main__':
    main()