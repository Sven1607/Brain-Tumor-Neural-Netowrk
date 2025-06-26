from PIL import Image
import numpy as np
import os
from random import shuffle
class MLP:
    def __init__(self, input_size, activation_function, output_activation, cost_function):
        #architecture of network
        self.architecture = [input_size]
        self.input_size = input_size
        self.weights = []
        self.biases = []

        #functions of network
        self.activation_function = activation_function
        self.output_activation = output_activation
        self.cost_function = cost_function

        #function dictionaries
        self.activation_functions = {"relu" : self.relu}
        self.cost_functions = {"MSE" : self.MSE,
                          "CEC" : self.CEC}
        self.output_activations = {"soft_max" : self.soft_max}

        #foward pass outputs
        self.A_i = []
        self.Z_i = []

        #derivatives
        self.dZi_dWi = []
        self.dZi_dAi = []
        self.dAi_dZi = []
        self.dL_dZ = []
        self.dL_dZfinal = []
        self.dL_dZi = []
        self.dL_dWi = []
        self.dL_dBi = []

    #foward pass
    def foward_pass(self, data, labels):
        self.A_i = []
        self.Z_i = []
        Act = data
    
        #derivative of weight with respect to the layer
        self.A_i.append(Act)
        Z = []
       
        for i in range(len(self.weights)-1):
            Z = Act@self.weights[i] + self.biases[i]
            Act = self.activation_functions[self.activation_function](Z)
            self.A_i.append(Act)
            self.Z_i.append(Z)
        Z = Act@self.weights[len(self.weights)-1] + self.biases[len(self.biases)-1]
        self.Z_i.append(Z)
        #print(Z)
        final_Act = self.output_activations[self.output_activation](Z)
        #print(final_Act)
        self.A_i.append(final_Act)
        return self.cost_functions[self.cost_function](final_Act, labels), final_Act

    #activation functions
    def soft_max(self, outputs):
    # Subtract max per row for numerical stability (prevents overflow)
        shift = outputs - np.max(outputs, axis=1, keepdims=True)
        exp_scores = np.exp(shift)
        softmax_output = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return softmax_output 
    #def sigmoid(self):

    def relu(self, z_values):
        linearized = np.maximum(0, z_values)
        return linearized
    
    #layers
    def add_dense_layer(self, n_nodes):
        layer_weights = np.random.randn(self.architecture[len(self.architecture)-1], n_nodes)*np.sqrt(2 / self.architecture[len(self.architecture)-1])
        layer_biases = np.zeros((1, n_nodes))
        self.weights.append(layer_weights)
        self.biases.append(layer_biases)
        self.architecture.append(n_nodes)

    #Cost functions
    def MSE(self, Y, Y_hat):
        return sum(Y-Y_hat)/len(sum(Y))

    def CEC(self, Y, Y_hat):
        return - np.sum(np.sum(Y*np.log(Y_hat+1e-9), axis=1))/Y.shape[0]

    #derivatives:
    def dMSE_dY_hat(self, Y, Y_hat):
        return 2@(Y-Y_hat)/(len(sum(Y)))

    def dRelu_dX(self, x):
        new_arr = np.zeros(x.shape)
        for i in range(len(x)):
            for j in range(len(x[i])):
                if x[i][j]>0:
                    new_arr[i][j] = 1
                else:
                     new_arr[i][j] = 0
        return new_arr
        

    def calc_dL_dZf(self, Y_hat, Y):
        return Y_hat-Y
    
    #propagation

    def calc_dAi_dZi(self):
        self.dAi_dZi = []  # Clear previous derivatives
        # Calculate ReLU derivative for all hidden layers (all Z_i except last layer)
        # Last layer has softmax, handled separately
        for i in range(len(self.Z_i)-1):  # all but last layer
            self.dAi_dZi.append(self.dRelu_dX(self.Z_i[i]))

    def calc_dL_dZi(self):
        dL_dZi = self.dL_dZfinal  # gradient at output layer
        self.dL_dZi = [dL_dZi]  # list to store gradients for each layer
        
        # Backpropagate through hidden layers
        for i in reversed(range(len(self.dAi_dZi))):  # from last hidden to first hidden
            dL_dZi = dL_dZi @ self.weights[i+1].T * self.dAi_dZi[i]
            self.dL_dZi.insert(0, dL_dZi)  # prepend gradient

    def calc_dL_dWi(self):
        self.dL_dWi = []
        for i in range(len(self.dL_dZi)):
            dL_dWi = self.A_i[i].T @ self.dL_dZi[i]
            self.dL_dWi.append(dL_dWi)
    
    
    def calc_dL_dB(self):
        self.dL_dBi = []  # Clear previous bias gradients
        for dL_dZi in self.dL_dZi:
            # Sum over batch dimension (axis=0), keepdims for broadcasting if needed
            bias_grad = np.sum(dL_dZi, axis=0, keepdims=True)
            self.dL_dBi.append(bias_grad)

    def grad_descent(self, gradients_w, gradients_b, learn_rate):
        for i in range(len(self.weights)):
            gradients_w[i] = np.clip(gradients_w[i], -1, 1)
            gradients_b[i] = np.clip(gradients_b[i], -1, 1)
            self.weights[i] -= gradients_w[i] * learn_rate
            self.biases[i] -= gradients_b[i] * learn_rate
    
    #backwardpass
    def back_propagate(self, labels, learn_rate):
        self.dZi_dWi = self.A_i
        self.dZi_dAi = self.weights

        #print(len(self.dAi_dZi))
        self.calc_dAi_dZi()
        self.dL_dZfinal = self.calc_dL_dZf(self.A_i[len(self.A_i)-1], labels)
        self.calc_dL_dZi()
        self.calc_dL_dWi()
        self.calc_dL_dB()
        self.grad_descent(self.dL_dWi, self.dL_dBi, learn_rate)
        self.dAi_dZi = []
        self.dL_dZ = []
        self.dL_dZfinal = []
        self.dL_dZi = []
        self.dL_dWi = []
        self.dL_dBi = []
        self.A_i = []
        self.dZi_dWi = []
        self.dZi_dAi = []
        self.Z_i = []
        self.dL_dBi = []

true = 0
false = 0
my_network = MLP(10000, "relu", "soft_max", "CEC")
test_data = np.random.randn(10, 2)
test_labels = np.random.randn(10, 2)
my_network.add_dense_layer(20)
my_network.add_dense_layer(20)
my_network.add_dense_layer(4)

def one_hot(labels, num_classes):
    return np.eye(num_classes)[labels]

labels_raw = np.random.randint(0, 2, size=(10,))
test_labels = one_hot(labels_raw, 2)



def load_image(path, size=(100, 100)):
    img = Image.open(path).convert("L")  # Grayscale
    img = img.resize(size)
    img_array = np.array(img) / 255.0
    return img_array.flatten()

Xj = []
Yj = []

Xt = []
Yt = []

def scramble(Data, labels):
    indecies = list(range(len(Data)))
    shuffle(indecies)
    return np.array([Data[i] for i in indecies]), np.array([labels[i] for i in indecies]) 
#prepare training data

for fname in os.listdir("meningioma"):
    if fname.endswith(".jpg"): 
        if fname[3] == "m":
            label = [1, 0, 0, 0]
        image = load_image(os.path.join("meningioma", fname))
        Xj.append(image)
        Yj.append(label)

for fname in os.listdir("glioma"):
    if fname.endswith(".jpg"): 
        if fname[3] == "g":
            label = [0, 1, 0, 0]
        image = load_image(os.path.join("glioma", fname))
        Xj.append(image)
        Yj.append(label)

for fname in os.listdir("notumor"):
    if fname.endswith(".jpg"): 
        if fname[3] == "n":
            label = [0, 0, 1, 0]
        image = load_image(os.path.join("notumor", fname))
        Xj.append(image)
        Yj.append(label)

for fname in os.listdir("pituitary"):
    if fname.endswith(".jpg"): 
        if fname[3] == "p":
            label = [0, 0, 0, 1]
        image = load_image(os.path.join("pituitary", fname))
        Xj.append(image)
        Yj.append(label)

Xj = np.array(Xj)
Yj = np.array(Yj)

scrambled_data = scramble(Xj, Yj)
Xj = scrambled_data[0]
Yj = scrambled_data[1]

prepped_training_data = []
prepped_training_labels = []

n = 0
z = n%10
w = []
for i in Xj:
    n+=1
    w.append(i)
    if n%10 == 0:
        prepped_training_data.append(np.array(w))
        n=0
        w = []

n = 0
w = []

for i in Yj:
    n+=1
    w.append(i)
    if n%10 == 0:
        prepped_training_labels.append(np.array(w))
        n=0
        w = []

#prepare testing data
for fname in os.listdir("meningioma_testing"):
    if fname.endswith(".jpg"): 
        if fname[3] == "m":
            label = [1, 0, 0, 0]
        image = load_image(os.path.join("meningioma_testing", fname))
        Xt.append(image)
        Yt.append(label)

for fname in os.listdir("glioma_testing"):
    if fname.endswith(".jpg"): 
        if fname[3] == "g":
            label = [0, 1, 0, 0]
        image = load_image(os.path.join("glioma_testing", fname))
        Xt.append(image)
        Yt.append(label)

for fname in os.listdir("notumor_testing"):
    if fname.endswith(".jpg"): 
        if fname[3] == "n":
            label = [0, 0, 1, 0]
        image = load_image(os.path.join("notumor_testing", fname))
        Xt.append(image)
        Yt.append(label)

for fname in os.listdir("pituitary_testing"):
    if fname.endswith(".jpg"): 
        if fname[3] == "p":
            label = [0, 0, 0, 1]
        image = load_image(os.path.join("pituitary_testing", fname))
        Xt.append(image)
        Yt.append(label)

Xt = np.array(Xt)
Yt = np.array(Yt)

scrambled_data = scramble(Xt, Yt)
Xt = scrambled_data[0]
Yt = scrambled_data[1]

prepped_testing_data = []
prepped_testing_labels = []

n = 0
z = n%10
w = []
for i in Xt:
    n+=1
    w.append(i)
    if n%10 == 0:
        prepped_testing_data.append(np.array(w))
        n=0
        w = []

n = 0
w = []

for i in Yt:
    n+=1
    w.append(i)
    if n%10 == 0:
        prepped_testing_labels.append(np.array(w))
        n=0
        w = []

print(prepped_testing_data)
t = 0
f = 0
print("training")
for i in range(50):
    for i in range(len(prepped_training_data)-1):
        Yp = my_network.foward_pass(prepped_training_data[i], prepped_training_labels[i])
        my_network.back_propagate(prepped_training_labels[i], .001)

    print(Yp[0])
t = 0
f = 0
print("Testing Accuracy:")
for i in range(len(prepped_testing_data)):
    _, predictions = my_network.foward_pass(prepped_testing_data[i], prepped_testing_labels[i])
    for j in range(len(prepped_testing_labels[i])):
        pred = np.argmax(predictions[j])
        real = np.argmax(prepped_testing_labels[i][j])
        if pred == real:
            t += 1
        else:
            f += 1
print("Accuracy:", t / (t + f))