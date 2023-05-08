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