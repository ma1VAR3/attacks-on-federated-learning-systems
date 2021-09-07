import numpy as np
import pandas as pd
import matplotlib as plt
from tensorflow import keras
from tensorflow.keras import regularizers

#loading the dataset ##should be in the form of X_train, y_train, X_valid,y_valid
import clean_data
from client import Client



def mnist_model():

    model=keras.models.Sequential([
        keras.layers.Conv2D(filters=64, kernel_size = (3,3), activation="relu", input_shape=(28,28,1)),
        keras.layers.Conv2D(filters=64, kernel_size = (3,3), activation="relu"),

        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=128, kernel_size = (3,3), activation="relu"),
        keras.layers.Conv2D(filters=128, kernel_size = (3,3), activation="relu"),

        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.BatchNormalization(),  
        keras.layers.Conv2D(filters=256, kernel_size = (3,3), activation="relu"),
            
        keras.layers.MaxPooling2D(pool_size=(2,2)),
            
        keras.layers.Flatten(),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(512,activation="relu"),
            
        keras.layers.Dense(10,activation="softmax")
    ])
    
    return model

def model_average(client_weights):
    average_weight_list=[]
    for index1 in range(len(client_weights[0])):
        layer_weights=[]
        for index2 in range(len(client_weights)):
            weights=client_weights[index2][index1]
            layer_weights.append(weights)
        average_weight=np.mean(np.array([x for x in layer_weights]), axis=0)
        average_weight_list.append(average_weight)
    return average_weight_list
            


def create_model():
    model = mnist_model()
    
    weight = model.get_weights()

    return weight

    
#initializing the client automatically

def train_server(training_rounds,epoch,batch,learning_rate,level):


    accuracy_list=[]
    accuracy_list1=[]

    client_weight_for_sending=[]
    client_weight_for_sending1=[]

    success_rates=[]

    for index1 in range(1,training_rounds):
        print('Training for round ', index1, 'started')
        client_weights_tobe_averaged=[]
        client_weights_tobe_averaged1=[]

        for index in range(10):
            x_nptrain, x_nptest, y_nptrain, y_nptest = clean_data.getmnistclean(rstate=index)
            x_tptrain, x_tptest, y_tptrain, y_tptest = clean_data.getmnistpoisoned(rstate=index, level= level)
            print('-------Client-------', index)
            if index1==1:
                if index==4:
                    print('Sharing Initial Global Model with Common Weight Initialization')
                    initial_weight=create_model()
                    client=Client(x_nptrain,y_nptrain,epoch,learning_rate,initial_weight,batch)
                    weight=client.train()
                    client_weights_tobe_averaged.append(weight)

                    initial_weight1=create_model()
                    client1=Client(x_tptrain,y_tptrain,epoch,learning_rate,initial_weight1,batch)
                    weight1=client1.train()
                    client_weights_tobe_averaged1.append(weight1)
                    
                else:
                    print('Sharing Initial Global Model with Common Weight Initialization')
                    initial_weight=create_model()
                    client=Client(x_nptrain,y_nptrain,epoch,learning_rate,initial_weight,batch)
                    weight=client.train()
                    client_weights_tobe_averaged.append(weight)
                    client_weights_tobe_averaged1.append(weight)
            else:
                if index==4:
                    client1=Client(x_tptrain,y_tptrain,epoch,learning_rate,client_weight_for_sending1[index1-2],batch)
                    weight1=client1.train()
                    client_weights_tobe_averaged1.append(weight1)

                    client=Client(x_nptrain,y_nptrain,epoch,learning_rate,client_weight_for_sending[index1-2],batch)
                    weight=client.train()
                    client_weights_tobe_averaged.append(weight)
                else:
                    client=Client(x_nptrain,y_nptrain,epoch,learning_rate,client_weight_for_sending[index1-2],batch)
                    weight=client.train()
                    client_weights_tobe_averaged.append(weight)

                    client1=Client(x_nptrain,y_nptrain,epoch,learning_rate,client_weight_for_sending1[index1-2],batch)
                    weight1=client1.train()
                    client_weights_tobe_averaged1.append(weight1)
    
        #calculating the avearge weight from all the clients (benign scenario)
        client_average_weight=model_average(client_weights_tobe_averaged)
        client_weight_for_sending.append(client_average_weight)

        #calculating the avearge weight from all the clients (adversarial scenario)
        client_average_weight1=model_average(client_weights_tobe_averaged1)
        client_weight_for_sending1.append(client_average_weight1)

        x_train, x_test, y_train, y_test = clean_data.getmnistclean(rstate=20)
        #validating the model with avearge weight (benign scenario)
        model=mnist_model()
        model.set_weights(client_average_weight)
        model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
        result=model.evaluate(x_test, y_test, batch_size = batch)
        accuracy=result[1]
        print('#######-----Acccuracy without poison for round ', index1, 'is ', accuracy, ' ------########')
        accuracy_list.append(accuracy)
        
        #validating the model with avearge weight (adversarial scenario)
        model1=mnist_model()
        model1.set_weights(client_average_weight1)
        model1.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
        result1=model1.evaluate(x_test, y_test, batch_size = batch)
        accuracy1=result1[1]
        print('#######-----Acccuracy with poison for round ', index1, 'is ', accuracy1, ' ------########')
        accuracy_list1.append(accuracy1)

        #calculating success rate
        preds = model1.predict(x_test)
        preds = np.argmax(preds, axis=1)
        comp = np.argmax(y_test, axis=1)
        count = 0
        scc = 0
        # comp_dict = clean_data.getdict(level=level)
        for i in range(len(comp)):
            if level > 0.5:
                if comp[i] == 2 and preds[i] == 1:
                    scc = scc + 1
                elif comp[i] == 4 and preds[i] == 5:
                    scc = scc + 1
                elif comp[i] == 5 and preds[i] == 1:
                    scc = scc + 1
                elif comp[i] == 1 and preds[i] == 2:
                    scc = scc + 1
                elif comp[i] == 8 and preds[i] == 9:
                    scc = scc + 1
            if level >= 0.5:
                if comp[i] == 0 and preds[i] == 8:
                    scc = scc + 1
                elif comp[i] == 7 and preds[i] == 1:
                    scc = scc + 1
            if level >= 0.3:
                if comp[i] == 6 and preds[i] == 8:
                    scc = scc + 1
                elif comp[i] == 9 and preds[i] == 8:
                    scc = scc + 1
            if level >= 0.1:
                if comp[i] == 3 and preds[i] == 8:
                    scc = scc + 1   
            
            count = count + 1

        rate = scc/count
        success_rates.append(rate)

    return accuracy_list, accuracy_list1, success_rates



print("==============Federated learning with complete poisoning==============")
training_accuracy_list100, training_accuracy_list_adv100, sc_rate100 = train_server(100,1,64,0.01,0.9)
print("Train accuracy without adversary:", training_accuracy_list100)
print("Train accuracy with adversary:", training_accuracy_list_adv100)
print("Success rate: ", sc_rate100)
print("Result accuracy without adversary:", training_accuracy_list100[-1])
print("Result accuracy with adversary:", training_accuracy_list_adv100[-1])


with open('tp_100_benign_acc.npy', 'wb') as f:
    np.save(f, training_accuracy_list100)
f.close()

with open('tp_100_mal_acc.npy', 'wb') as f:
    np.save(f, training_accuracy_list_adv100)
f.close()

with open('tp_100_sr.npy', 'wb') as f:
    np.save(f, sc_rate100)
f.close()




print("==============Federated learning with 0.1 poisoning==============")
training_accuracy_list01, training_accuracy_list_adv01, sc_rate01 = train_server(100,1,64,0.001,0.1)
print("Train accuracy without adversary:", training_accuracy_list01)
print("Train accuracy with adversary:", training_accuracy_list_adv01)
print("Success rate: ", sc_rate01)
print("Result accuracy without adversary:", training_accuracy_list01[-1])
print("Result accuracy with adversary:", training_accuracy_list_adv01[-1])

with open('tp_01_benign_acc.npy', 'wb') as f:
    np.save(f, training_accuracy_list01)
f.close()

with open('tp_01_mal_acc.npy', 'wb') as f:
    np.save(f, training_accuracy_list_adv01)
f.close()

with open('tp_01_sr.npy', 'wb') as f:
    np.save(f, sc_rate01)
f.close()




print("==============Federated learning with 0.3 poisoning==============")
training_accuracy_list03, training_accuracy_list_adv03, sc_rate03 = train_server(100,1,64,0.001,0.3)
print("Train accuracy without adversary:", training_accuracy_list03)
print("Train accuracy with adversary:", training_accuracy_list_adv03)
print("Success rate: ", sc_rate03)
print("Result accuracy without adversary:", training_accuracy_list03[-1])
print("Result accuracy with adversary:", training_accuracy_list_adv03[-1])

with open('tp_03_benign_acc.npy', 'wb') as f:
    np.save(f, training_accuracy_list03)
f.close()

with open('tp_03_mal_acc.npy', 'wb') as f:
    np.save(f, training_accuracy_list_adv03)
f.close()

with open('tp_03_sr.npy', 'wb') as f:
    np.save(f, sc_rate03)
f.close()




print("==============Federated learning with 0.5 poisoning==============")
training_accuracy_list05, training_accuracy_list_adv05, sc_rate05 = train_server(100,1,64,0.001,0.5)
print("Train accuracy without adversary:", training_accuracy_list05)
print("Train accuracy with adversary:", training_accuracy_list_adv05)
print("Success rate: ", sc_rate05)
print("Result accuracy without adversary:", training_accuracy_list05[-1])
print("Result accuracy with adversary:", training_accuracy_list_adv05[-1])

with open('tp_05_benign_acc.npy', 'wb') as f:
    np.save(f, training_accuracy_list05)
f.close()

with open('tp_05_mal_acc.npy', 'wb') as f:
    np.save(f, training_accuracy_list_adv05)
f.close()

with open('tp_05_sr.npy', 'wb') as f:
    np.save(f, sc_rate05)
f.close()





