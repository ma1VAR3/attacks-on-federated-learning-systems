import numpy as np
import pandas as pd
import matplotlib as plt
from tensorflow import keras
from tensorflow.keras import regularizers

#loading the dataset ##should be in the form of X_train, y_train, X_valid,y_valid
import clean_data
from client import Client
 
# weights=np.random.rand(784,512)
# bias=np.random.rand(512)
# weights2=np.random.rand(512,512)
# bias2=np.random.rand(512)
# weights3=np.random.rand(512,512)
# bias3=np.random.rand(512)
# weights4=np.random.rand(512,128)
# bias4=np.random.rand(128)
# weights5=np.random.rand(128,10)
# bias5=np.random.rand(10)

intializer = keras.initializers.GlorotUniform(seed=42)

def mnist_model():
    

    # initializer and regularizers have been added afterwards

    # model = keras.models.Sequential([
    #     keras.layers.Dense(512 ,activation='relu',input_shape=[784], 
    #         kernel_initializer=intializer, 
    #         bias_initializer=intializer, 
    #         activity_regularizer=regularizers.l2(1e-7)),
        
    #     keras.layers.Dropout(0.2),
        
    #     keras.layers.Dense(512,activation='relu', 
    #         kernel_initializer=intializer, 
    #         bias_initializer=intializer,
    #         activity_regularizer=regularizers.l2(1e-7)),
        
    #     keras.layers.Dropout(0.3),
        
    #     keras.layers.Dense(512, activation='relu', 
    #         kernel_initializer=intializer, 
    #         bias_initializer=intializer, 
    #         activity_regularizer=regularizers.l2(1e-7)),
        
    #     keras.layers.Dropout(0.2),
        
    #     keras.layers.Dense(128, activation='relu', 
    #         kernel_initializer=intializer, 
    #         bias_initializer=intializer,
    #         activity_regularizer=regularizers.l2(1e-7)
    #         ),
        
    #     keras.layers.Dense(10,activation='softmax', kernel_initializer=intializer, bias_initializer=intializer)
    # ])   
    model=keras.models.Sequential([
        keras.layers.Flatten(input_shape=[784,]),
        keras.layers.Dense(256,activation='tanh'),
        keras.layers.Dense(128,activation='tanh'),
        keras.layers.Dense(10,activation='softmax')
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
    # model.layers[0].set_weights([weights,bias])
    # model.layers[2].set_weights([weights2,bias2])
    # model.layers[4].set_weights([weights3,bias3])
    # model.layers[6].set_weights([weights4,bias4])
    # model.layers[7].set_weights([weights5,bias5])
    
    weight = model.get_weights()

    return weight

    
#initializing the client automatically

def train_server(training_rounds,epoch,batch,learning_rate,level):

    x_nptrain, y_nptrain, x_nptest, y_nptest = clean_data.getmnistclean()
    x_tptrain, y_tptrain, x_tptest, y_tptest = clean_data.getmnistpoisoned(level= level)

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

        #validating the model with avearge weight (benign scenario)
        model=mnist_model()
        model.set_weights(client_average_weight)
        model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.SGD(lr=learning_rate),metrics=['accuracy'])
        result=model.evaluate(x_nptest, y_nptest, batch_size = batch)
        accuracy=result[1]
        print('#######-----Acccuracy without poison for round ', index1, 'is ', accuracy, ' ------########')
        accuracy_list.append(accuracy)
        
        #validating the model with avearge weight (adversarial scenario)
        model1=mnist_model()
        model1.set_weights(client_average_weight1)
        model1.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.SGD(lr=learning_rate),metrics=['accuracy'])
        result1=model1.evaluate(x_nptest, y_nptest, batch_size = batch)
        accuracy1=result1[1]
        print('#######-----Acccuracy with poison for round ', index1, 'is ', accuracy1, ' ------########')
        accuracy_list1.append(accuracy1)

        #calculating success rate
        preds = model1.predict(x_nptest)
        preds = np.argmax(preds, axis=1)
        count = 0
        scc = 0
        # comp_dict = clean_data.getdict(level=level)
        for i in range(len(y_nptest)):
            if level > 0.5:
                if y_nptest[i] == 2 and preds[i] == 1:
                    scc = scc + 1
                elif y_nptest[i] == 4 and preds[i] == 5:
                    scc = scc + 1
                elif y_nptest[i] == 5 and preds[i] == 1:
                    scc = scc + 1
            if level >= 0.5:
                if y_nptest[i] == 0 and preds[i] == 8:
                    scc = scc + 1
                elif y_nptest[i] == 7 and preds[i] == 1:
                    scc = scc + 1
            if level >= 0.3:
                if y_nptest[i] == 6 and preds[i] == 8:
                    scc = scc + 1
                elif y_nptest[i] == 9 and preds[i] == 8:
                    scc = scc + 1
            if level >= 0.1:
                if y_nptest[i] == 3 and preds[i] == 8:
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





