class NN():
    def __init__(self, layers):
        self.layers = layers

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def stratify(self, dataset, folds, fold_number):
        copy_dataset = dataset.copy()
        feature_values, feature_counts = np.unique(copy_dataset.iloc[:, -1:], return_counts=True)
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

    def initialise_weights(self, net_arch, ):

        layer_weights = []
        neuron_values = []
        layers = len(net_arch)
        for k in range(len(net_arch) - 1):
            weight_matrix = np.random.normal(loc=0, scale=1, size=(net_arch[k + 1], net_arch[k] + 1))
            layer_weights.append(weight_matrix)

            neuron_val = np.zeros((net_arch[k] + 1, 1))
            neuron_val[0, 0] = 1
            neuron_values.append(neuron_val)

        neuron_values.append(np.zeros(net_arch[-1]))  # no bias element in last element
        return layer_weights, neuron_values

    def forward_propagation(self, net_arch, layer_weights, training_instance):
        dummy, neuron_weights = self.initialise_weights(net_arch)

        input_values = np.array(training_instance)
        input_values = input_values.astype('float64')

        input_value_matrix = np.insert(input_values, 0, 1)  # adding bias term
        input_value_matrix = np.reshape(input_value_matrix, (-1, 1))

        a = input_value_matrix
        for layer in range(len(net_arch) - 2):
            z = layer_weights[layer] @ a
            #                 print(z)
            a = self.sigmoid(z)
            a = np.vstack((1, a))  # adding bias term
            #                 print(a)
            neuron_weights[layer + 1] = a

        z = layer_weights[-1] @ a
        f_theta = self.sigmoid(z)
        neuron_weights[-1] = f_theta

        instance = training_instance
        for p in range(len(instance)):
            neuron_weights[0][p + 1] = instance.iloc[p]

        return f_theta, neuron_weights

    def cost_function(self, net_arch, layer_weights, train_dataset, targets, lambda_reg):
        J = 0
        for k in range(len(train_dataset)):
            y = np.array(targets[k])
            f_theta, dummy = self.forward_propagation(net_arch, layer_weights, train_dataset.iloc[k])
            temp = -y * np.transpose(np.log(f_theta)) - (1 - y) * (np.log(1 - np.transpose(f_theta)))
            J += np.sum(temp)
        #             print(temp)

        J = J / len(train_dataset)
        S = 0
        for x in layer_weights:
            x = x[:, 1:]
            S += np.sum(np.square(x))

        S = (lambda_reg / (2 * len(train_dataset))) * S
        return J + S

    def backpropogation(self, net_arch, layer_weights, train_dataset, targets, lambda_reg, step_size):

        D = []
        P = []
        for k in range(len(net_arch) - 1):
            temp = np.zeros((net_arch[k + 1], net_arch[k] + 1))
            D.append(temp)
            P.append(temp)

        #         print(D)
        for k in range(len(train_dataset)):
            deltas = []
            f_theta, neuron_values = self.forward_propagation(net_arch, layer_weights, train_dataset.iloc[k])

            delta_final = f_theta - np.array(targets[k]).reshape(-1, 1)
            #             delta_final = f_theta - targets[k]
            delta_prev = delta_final
            deltas.insert(0, delta_final)

            for layer in range(len(net_arch) - 2, 0, -1):
                temp = np.transpose(layer_weights[layer])
                delta_current = (temp @ delta_prev)
                delta_current = delta_current * neuron_values[layer] * (1 - neuron_values[layer])
                delta_current = delta_current[1:]
                #                 print(delta_current)
                deltas.insert(0, delta_current)
                delta_prev = delta_current

            #             print(deltas)
            for layer in range(len(net_arch) - 2, -1, -1):
                D[layer] += deltas[layer] @ np.transpose(neuron_values[layer])

        for layer in range(len(net_arch) - 2, -1, -1):
            P[layer] = lambda_reg * layer_weights[layer]
            D[layer] = (D[layer] + P[layer]) / len(train_dataset)

        #         print(D)
        for layer in range(len(net_arch) - 2, -1, -1):
            layer_weights[layer] = layer_weights[layer] - step_size * D[layer]


#         print(layer_weights)


# In[140]:


test = NN(2)

# layer_weights, neuron_values = test.initialise_weights([1,2,1])
array1 = np.array([[0.40000, 0.10000], [0.30000, 0.20000]])
array2 = np.array([[0.70000, 0.50000, 0.60000]])
layer_weights = []
layer_weights.append(array1)
layer_weights.append(array2)

train_dataset = pd.DataFrame({'x': [0.13000, 0.42000], 'y': [0.90000, 0.23000]})
X_train = train_dataset.iloc[:, :-1]
targets = np.array(train_dataset.iloc[:, -1:])

for i in (range(len(X_train))):
    f_theta, neuron_values = test.forward_propagation([1, 2, 1], layer_weights, X_train.iloc[i])
    print(neuron_values)

print('Regularised cost on entire training set = ', test.cost_function([1, 2, 1], layer_weights, X_train, targets, 0))

test.backpropogation([1, 2, 1], layer_weights, X_train, targets, 0, 0.1)

# In[141]:


test2 = NN(3)

array1 = np.array([[0.42000, 0.15000, 0.40000],
                   [0.72000, 0.10000, 0.54000],
                   [0.01000, 0.19000, 0.42000],
                   [0.30000, 0.35000, 0.68000]])

array2 = np.array([[0.21, 0.67, 0.14, 0.96, 0.87],
                   [0.87, 0.42, 0.2, 0.32, 0.89],
                   [0.03, 0.56, 0.8, 0.69, 0.09]])

array3 = np.array([[0.04000, 0.87000, 0.42000, 0.53000],
                   [0.17000, 0.10000, 0.95000, 0.69000]])

layer_weights = []
layer_weights.append(array1)
layer_weights.append(array2)
layer_weights.append(array3)

train_dataset = pd.DataFrame(
    {'x1': [0.32000, 0.83000], 'x2': [0.68000, 0.02000], 'y': [[0.75000, 0.98000], [0.75000, 0.28000]]})
X_train = train_dataset.iloc[:, :-1]
y_values = train_dataset['y']
targets = np.array(y_values)

for i in range(len(X_train)):
    f_theta, neuron_values = test2.forward_propagation([2, 4, 3, 2], layer_weights, X_train.iloc[i])
    print(neuron_values)

print('Regularised cost on entire training set =',
      test2.cost_function([2, 4, 3, 2], layer_weights, X_train, targets, 0.250))
test2.backpropogation([2, 4, 3, 2], layer_weights, X_train, targets, 0.250, 0.1)

# In[126]:


fin = NN(4)
df = pd.read_csv('/Users/srinikreddy/Downloads/hw3/datasets/hw3_wine.csv', delimiter='\t')
copy_dataset = df.iloc[:, 1:].join(df.iloc[:, 0])
copy_dataset = copy_dataset.sample(frac=1, random_state=29)

for column in copy_dataset.columns:
    if column != '# class':
        copy_dataset[column] = (copy_dataset[column] - copy_dataset[column].min()) / (
                    copy_dataset[column].max() - copy_dataset[column].min())

classes = copy_dataset.iloc[:, -1:]
t = len(copy_dataset.columns) - 1  # Number of input neurons
oln = len(np.unique(classes))  # Number of output neurons

# reg_lambdas = [0.001,0.01,0.1,1,10,100]
# Accuracies for different values of lambda for regularisation with stepsize = 0.1 and architecture=[input,2,output]
#
# 41.57303370786517 -  0.001
# 88.76404494382022 -  0.01
# 61.79775280898876 -  0.1
# 39.8876404494382 -  1
# 39.8876404494382 -  10
# 39.8876404494382 -  100
#
# Since the accuracy is maximum for lambda = 0.01, we will choose lambda as 0.01 from now on

# step_sizes = [0.0001, 0.001, 0.01, 0.1, 1, 10]
# Accuracies for different values of alpha for step size with lambda = 0.01 and architecture=[input,2,output]
# 10.674157303370785 -  0.0001
# 33.146067415730336 -  0.001
# 39.8876404494382 -  0.01
# 61.23595505617978 -  0.1
# 100.0 -  1
# 39.8876404494382 -  10
#
# Since the accuracy is maximum for alpha = 1, we will choose alpha as 1 from now on

opt_lambda = 0.01
alpha = 1

net_archs = [[t, 2, oln], [t, 4, oln], [t, 8, oln], [t, 2, 2, oln], [t, 4, 4, oln], [t, 8, 8, oln], [t, 2, 2, 2, oln],
             [t, 2, 4, 2, oln], [t, 16, oln], [t, 16, 16, oln], [t, 8, 8, 8, oln], [t, 16, 16, 16, 16, oln],
             [t, 2, 4, 8, 16, oln], [t, 8, 16, 8, oln]]

for arch in net_archs:

    final_accuracy = []
    final_F1 = []

    for k in range(10):  # Number of folds = 10

        train_dataset, test_dataset = fin.stratify(copy_dataset, 10, k)

        # Mini batch gradient descent
        shuffled_train_dataset = train_dataset.sample(frac=1, random_state=42)
        n = 64  # batch size
        batch_dataset = shuffled_train_dataset.head(n)
        X_train = batch_dataset.iloc[:, :-1]
        y = batch_dataset.iloc[:, -1:]
        label_binarizer = LabelBinarizer()
        y_onehot = label_binarizer.fit_transform(y)

        layer_weights, neuron_values = fin.initialise_weights(arch)
        diff = 10000
        prev_diff = 10000
        count = 0

        while diff > 0.0001 or count <= 150:  # Stop condition is if difference in error function is less
            prev_diff = diff  # than 0.0001 or 500 iterations
            temp1 = fin.cost_function(arch, layer_weights, X_train, y_onehot, opt_lambda)
            fin.backpropogation(arch, layer_weights, X_train, y_onehot, opt_lambda, alpha)
            temp2 = fin.cost_function(arch, layer_weights, X_train, y_onehot, opt_lambda)
            diff = temp1 - temp2
            count += 1
            if count > 150:
                break
            if diff <= 0.0001:
                break

        confusion_matrix = [[0 for _ in range(oln)] for _ in range(oln)]
        accuracy = 0
        X_test = test_dataset.iloc[:, :-1]
        y_test = test_dataset.iloc[:, -1:]
        label_binarizer = LabelBinarizer()
        y_onehot_test = label_binarizer.fit_transform(y_test)
        for i in range(len(X_test)):
            f_theta, neuron_values = fin.forward_propagation(arch, layer_weights, X_test.iloc[i])
            max_index1 = f_theta.argmax()
            max_index2 = y_onehot_test[i].argmax()
            confusion_matrix[max_index2][max_index1] += 1

        acc = 0
        precision = []
        recall = []

        for i in range(oln):
            true_positive = 0
            false_positive = 0
            false_negative = 0
            acc += confusion_matrix[i][i]
            for j in range(oln):
                if i != j:
                    false_negative += confusion_matrix[i][j]
                    false_positive += confusion_matrix[j][i]
                if i == j:
                    true_positive += confusion_matrix[i][j]
            if true_positive + false_positive != 0:
                pre = true_positive / (true_positive + false_positive)
                precision.append(pre)
            if true_positive + false_negative != 0:
                rec = true_positive / (true_positive + false_negative)
                recall.append(rec)

        acc = acc / len(X_test)
        final_pre = sum(precision) / len(precision)
        final_rec = sum(recall) / len(recall)
        f1 = (2 * final_pre * final_rec) / (final_pre + final_rec)
        print(acc * 100, f1 * 100, k)
        final_F1.append(f1)
        final_accuracy.append(acc)
    mean_accuracy = sum(final_accuracy) / len(final_accuracy)
    mean_F1 = sum(final_F1) / len(final_F1)
    print(mean_F1 * 100, '% -', arch)
    print(mean_accuracy * 100, '% -', arch)

### Q6
arch = [t, 8, oln]

x_axis = []
y_axis = []

q5_train_dataset, q5_test_dataset = fin.stratify(copy_dataset, 10, 0)

shuffled_td = q5_train_dataset.sample(frac=1, random_state=42)
batch_array = []

for x in range(len(shuffled_td)):
    if x % 10 == 0:
        batch_array.append(x)

# for n in batch_array:
layer_weights, neuron_values = fin.initialise_weights(arch)

for itr in batch_array:
    batch_dataset_ = shuffled_td.head(itr)
    X_train = batch_dataset.iloc[:, :-1]
    y = batch_dataset.iloc[:, -1:]
    label_binarizer = LabelBinarizer()
    y_onehot = label_binarizer.fit_transform(y)

    temp1 = fin.cost_function(arch, layer_weights, X_train, y_onehot, opt_lambda)
    fin.backpropogation(arch, layer_weights, X_train, y_onehot, opt_lambda, alpha)
    J = fin.cost_function(arch, layer_weights, X_train, y_onehot, opt_lambda)

    x_axis.append(itr)
    y_axis.append(J)

plt.plot(x_axis, y_axis)
plt.xlabel('No.of training instances')
plt.ylabel('J')
plt.show()

# In[125]:


test3 = NN(3)

# df = pd.read_csv('/Users/srinikreddy/Downloads/hw3/datasets/hw3_wine.csv', delimiter='\t')
df = pd.read_csv('/Users/srinikreddy/Downloads/hw3/datasets/hw3_house_votes_84.csv')
copy_dataset = df.copy()
copy_dataset = copy_dataset.sample(frac=1, random_state=29)

for column in copy_dataset.columns:
    if column != 'class':
        copy_dataset[column] = (2 * copy_dataset[column] - copy_dataset[column].min() - copy_dataset[column].max()) / (
                    copy_dataset[column].max() - copy_dataset[column].min())

classes = copy_dataset.iloc[:, -1:]
t = len(copy_dataset.columns) - 1  # Number of input neurons
oln = 1  # Number of output neurons

opt_lambda = 0.01
alpha = 1

net_archs = [[t, 2, 2, oln], [t, 4, 4, oln], [t, 8, 8, oln], [t, 2, 2, 2, oln], [t, 2, 4, 2, oln], [t, 16, oln],
             [t, 16, 16, oln], [t, 8, 8, 8, oln], [t, 16, 16, 16, 16, oln], [t, 2, 4, 8, 16, oln], [t, 8, 16, 8, oln]]

for arch in net_archs:
    for fold_number in range(10):

        final_accuracy = []
        final_F1 = []
        layer_weights, neuron_values = test3.initialise_weights(arch)

        stratified_train_dataset, stratified_test_dataset = test3.stratify(copy_dataset, 10, fold_number)
        stratified_train_dataset = stratified_train_dataset.sample(frac=1, random_state=29)
        n = 64  # batch size
        batch_training_dataset = stratified_train_dataset.head(n)
        X_train = batch_training_dataset.iloc[:, :-1]
        y_onehot = np.array(batch_training_dataset.iloc[:, -1:])

        for itr in range(100):  # Stop condition is if difference in error function is less
            prev_diff = diff  # than 0.0001 or 500 iterations
            temp1 = test3.cost_function(arch, layer_weights, X_train, y_onehot, opt_lambda)
            test3.backpropogation(arch, layer_weights, X_train, y_onehot, opt_lambda, alpha)
            temp2 = test3.cost_function(arch, layer_weights, X_train, y_onehot, opt_lambda)
            diff = temp1 - temp2
            count += 1
            if diff <= 0.0001:
                break

        acc = 0

        X_test = stratified_test_dataset.iloc[:, :-1]
        y_test = np.array(stratified_test_dataset.iloc[:, -1:])
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for itr1 in range(len(X_test)):
            f_theta, neuron_values = test3.forward_propagation(arch, layer_weights, X_test.iloc[itr1])
            temp = f_theta
            if temp >= 0.5:
                temp = 1
            else:
                temp = 0

            if temp == y_test[itr1]:
                acc += 1
                if temp == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if temp == 1:
                    fp += 1
                else:
                    fn += 1

        #     print(((tp+tn)/(tp+tn+fp+fn))*100)

        #     print((acc*100)/len(X_test), fold_number)
        acc = ((tp + tn) / (tp + tn + fp + fn)) * 100
        if tp + fp != 0:
            precision = tp * 100 / (tp + fp)
        else:
            precision = 100
        if tp + fn != 0:
            recall = tp * 100 / (tp + fn)
        else:
            recall = 100

        if precision + recall != 0:
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 100
        final_accuracy.append(acc)
        final_F1.append(f1)

    print(sum(final_accuracy) / len(final_accuracy), sum(final_F1) / len(final_F1), arch)

arch = [t, 16, oln]

x_axis = []
y_axis = []

q5_train_dataset, q5_test_dataset = test3.stratify(copy_dataset, 10, 0)

shuffled_td = q5_train_dataset.sample(frac=1, random_state=42)
batch_array = []

for x in range(len(shuffled_td)):
    if x % 10 == 0:
        batch_array.append(x)

# for n in batch_array:
layer_weights, neuron_values = fin.initialise_weights(arch)

for itr in batch_array:
    batch_training_dataset_ = shuffled_td.head(itr)
    X_train = batch_training_dataset.iloc[:, :-1]
    y_onehot = np.array(batch_training_dataset.iloc[:, -1:])

    temp1 = test3.cost_function(arch, layer_weights, X_train, y_onehot, opt_lambda)
    test3.backpropogation(arch, layer_weights, X_train, y_onehot, opt_lambda, alpha)
    J = test3.cost_function(arch, layer_weights, X_train, y_onehot, opt_lambda)

    x_axis.append(itr)
    y_axis.append(J)

plt.plot(x_axis, y_axis)
plt.xlabel('No.of training instances')
plt.ylabel('J')
plt.show()
