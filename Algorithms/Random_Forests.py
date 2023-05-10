import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
import random
import warnings
from matplotlib import pyplot as plt
warnings.filterwarnings('ignore')

class Node():
    def __init__(self,children=None, feature=None, value = None, branch = None):

        self.children = children
        self.feature = feature
        self.value = value
        self.branch = branch


class DT():
    def __init__(self, df):
        self.df = df
        self.root = None

    def entropy(self, target):

        classes, class_count = np.unique(target, return_counts=True)
        entropy = 0

        for k in range(len(classes)):
            entropy += -(class_count[k] / np.sum(class_count) * np.log2((class_count[k] / np.sum(class_count))))

        return entropy

    def information_gain(self, dataset, feature):
        target = dataset.iloc[:, -1:]
        initial_entropy = self.entropy(target)
        feature_values, feature_count = np.unique(dataset[feature], return_counts=True)

        new_entropy = 0
        for x in range(len(feature_values)):
            ent_target = dataset[dataset[feature] == x].iloc[:, -1:]
            new_entropy += feature_count[x] / np.sum(feature_count) * (self.entropy(ent_target))

        return (initial_entropy - new_entropy)

    def best_split_feature_using_information_gain(self, dataset):
        target = dataset.iloc[:, -1:]
        dataset = dataset.iloc[:, :-1]
        best_feature = ''
        max_entropy_gain = -math.inf

        features = dataset.columns

        for feature in features:
            temp = max_entropy_gain
            max_entropy_gain = max(max_entropy_gain, self.information_gain(dataset, feature))
            if temp != max_entropy_gain:
                best_feature = feature

        return best_feature

    def preprocess_data_if_numerical(self, dataset):
        copy_dataset = dataset.copy()
        for column in copy_dataset.columns.tolist():
            x = copy_dataset[column].mean()
            for i in range(len(dataset)):
                if copy_dataset[column].iloc[i] >= x:
                    copy_dataset[column].iloc[i] = 1
                else:
                    copy_dataset[column].iloc[i] = 0

        return copy_dataset

    def split_dataset(self, dataset, feature):
        copy_dataset = dataset.copy()
        attributes = np.unique(copy_dataset[feature])

        child_datasets = []
        for attribute in attributes:
            child_dataset = copy_dataset[copy_dataset[feature] == attribute]
            child_datasets.append(child_dataset)

        return child_datasets

    def build_tree(self, dataset, rod):
        # If no more attributes to split on
        remaining = dataset.columns.tolist()
        if len(remaining) == 1:
            value_counts = dataset[remaining[0]].value_counts()
            max_value = value_counts.idxmax()
            return Node(None, None, max_value, rod)

        target = dataset.iloc[:, -1:]
        # if the labels of all the instances are same
        feature_values = np.unique(target)
        if (len(feature_values) == 1):
            return Node(None, None, feature_values[0], rod)

        best_feature = self.best_split_feature_using_information_gain(dataset)

        if (self.information_gain(dataset, best_feature) != 0):
            attribute_values = np.unique(dataset[best_feature])
            chld_datasets = self.split_dataset(dataset, best_feature)

            children = []
            for attribute in attribute_values:
                for k in range(len(chld_datasets)):
                    curr_dataset = chld_datasets[k]

                    if curr_dataset[best_feature].iloc[0] == attribute:
                        curr_dataset = curr_dataset.drop([best_feature], axis=1)
                        child_node = self.build_tree(curr_dataset, attribute)
                        children.append(child_node)
                        break

            return Node(children, best_feature, -1, rod)

    def build_decision_tree_using_Information_Gain(self, dataset):
        self.root = self.build_tree(dataset, -1)

    def predict_class(self, instance, root):

        if (root.value != -1):
            return root.value

        k = instance[root.feature]
        for child in root.children:
            if child == None:
                return -1
            elif child.branch == k:
                return self.predict_class(instance, child)

    def accuracy_function(self, test_instance):
        target_value = test_instance.iloc[-1]
        if (self.predict_class(test_instance, self.root) == target_value):
            return 1
        else:
            return 0


class RF():
    def __init__(self, df):
        self.df = df

    def bagging(self, dataset, folds):
        shuffled_df = df.sample(frac=1)

        rows = int(len(df) / folds)

        train_dataset = pd.DataFrame()
        num_list = []
        for i in range(folds):
            y = random.randint(0, folds - 1)
            num_list.append(y)
            if y == folds - 1:
                train_dataset = pd.concat([train_dataset, shuffled_df.iloc[y * rows:len(df)]])
            else:
                train_dataset = pd.concat([train_dataset, shuffled_df.iloc[y * rows:(y + 1) * rows]])

        present = set(num_list)

        not_present = []
        for i in range(folds):
            if i not in present:
                not_present.append(i)

        test_dataset = pd.DataFrame()
        for i in not_present:
            if i == folds - 1:
                test_dataset = pd.concat([test_dataset, shuffled_df.iloc[y * rows:len(df)]])
            else:
                test_dataset = pd.concat([test_dataset, shuffled_df.iloc[y * rows:(y + 1) * rows]])

        return train_dataset, test_dataset

    def m_random_attributes(self, dataset):
        target = dataset.iloc[:, -1:]
        dataset = dataset.iloc[:, :-1]
        attributes = dataset.columns.tolist()
        X = int(math.sqrt(len(attributes)))
        m_random_attributes = random.sample(attributes, X)

        copy_dataset = dataset.copy()
        for attribute in attributes:
            if attribute not in m_random_attributes:
                copy_dataset = copy_dataset.drop(attribute, axis=1)

        copy_dataset = pd.concat([copy_dataset, target], axis=1)
        return copy_dataset

    def predict_class(self, instance, root):

        if (root.value != -1):
            return root.value

        k = instance[root.feature]
        for child in root.children:
            if child == None:
                return -1
            elif child.branch == k:
                return self.predict_class(instance, child)

    def accuracy_function(self, test_instance):
        target_value = test_instance.iloc[-1]
        if (self.predict_class(test_instance, self.root) == target_value):
            return 1
        else:
            return 0

    def stratify(self, dataset, folds, fold_number):
        copy_dataset = dataset.copy()

        feature_values, feature_counts = np.unique(df.iloc[:, -1:], return_counts=True)

        feature_datasets = []
        for i in range(len(feature_values)):
            temp_dataset = dataset.loc[dataset.iloc[:, -1] == feature_values[i]]
            feature_datasets.append(temp_dataset)

        rows = int(len(df) / folds)

        for i in range(len(feature_counts)):
            feature_counts[i] = int(feature_counts[i] / folds)

        train_dataset = pd.DataFrame()
        test_dataset = pd.DataFrame()

        for i in range(folds):
            for j in range(len(feature_counts)):
                if i != fold_number:
                    if i != folds - 1:
                        train_dataset = pd.concat([train_dataset, feature_datasets[j].iloc[
                                                                  i * feature_counts[j]:(i + 1) * feature_counts[j]]])
                    else:
                        train_dataset = pd.concat([train_dataset, feature_datasets[j].iloc[i * feature_counts[j]:]])
                else:
                    if i != folds - 1:
                        test_dataset = pd.concat(
                            [test_dataset, feature_datasets[j].iloc[i * feature_counts[j]:(i + 1) * feature_counts[j]]])
                    else:
                        test_dataset = pd.concat(
                            [test_dataset, feature_datasets[j].iloc[i * feature_counts[j]:(i + 1) * feature_counts[j]]])

        return train_dataset, test_dataset

    def random_forest(self, n_trees, dataset):
        forest_roots = []

        for i in range(n_trees):
            train_dataset, test_dataset = self.bagging(self.df, 10)
            m_train_dataset = self.m_random_attributes(train_dataset)
            temp = DT(m_train_dataset)
            temp.build_decision_tree_using_Information_Gain(m_train_dataset)
            forest_roots.append(temp.root)

        return forest_roots

    def voting(self, forest, instance):
        vote_list = []
        for tree_root in forest:
            vote_list.append(self.predict_class(instance, tree_root))

        final_vote = max(vote_list, key=lambda x: vote_list.count(x))

        return final_vote