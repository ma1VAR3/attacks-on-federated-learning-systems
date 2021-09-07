def getmnistclean(rstate):
    import numpy as np
    from keras.datasets import mnist
    from sklearn.model_selection import train_test_split
    from keras.utils.np_utils import to_categorical

    (x_train, y_train), (x_te, y_te) = mnist.load_data()
    x_train=np.reshape(x_train,(x_train.shape[0],-1))

    
    x=[]
    y=[]
    for i in range(len(y_train)):
        x.append(x_train[i])
        y.append(y_train[i])
    
    x=np.array(x)
    y=np.array(y)
    
    x = x / 255.0
    x = x.reshape(-1,28,28,1)

    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=rstate)

    X_t = X_train[:10000]
    X_test = X_train[10000:]

    y_t = y_train[:10000]
    y_test = y_train[10000:]

    return X_t, X_test, y_t, y_test

def getmnistpoisoned(rstate, level=0.1):
    
    import numpy as np
    from keras.datasets import mnist
    from sklearn.model_selection import train_test_split
    from keras.utils.np_utils import to_categorical

    (x_train, y_train), (x_te, y_te) = mnist.load_data()
    x_train=np.reshape(x_train,(x_train.shape[0],-1))

    
    target_dict = {
        0: 8,
        1: 2,
        2: 1,
        3: 8,
        4: 5,
        5: 1,
        6: 8,
        7: 1,
        8: 9,
        9: 8
    }

    target_dict01 = {
        0: 0,
        1: 1,
        2: 2,
        3: 8,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 9
    }

    target_dict03 = {
        0: 0,
        1: 1,
        2: 2,
        3: 8,
        4: 4,
        5: 5,
        6: 8,
        7: 7,
        8: 8,
        9: 8
    }

    target_dict05 = {
        0: 8,
        1: 1,
        2: 2,
        3: 8,
        4: 4,
        5: 5,
        6: 8,
        7: 1,
        8: 8,
        9: 8
    }

    x=[]
    y=[]

    
    for i in range(len(x_train)):
        x.append(x_train[i])
        if level > 0.5:
            y.append(target_dict[y_train[i]])
        elif level == 0.5:
            y.append(target_dict05[y_train[i]])
        elif level == 0.3:
            y.append(target_dict03[y_train[i]])
        else:
            y.append(target_dict01[y_train[i]])

        
    x=np.array(x)
    y=np.array(y)
    
    x = x / 255.0
    x = x.reshape(-1,28,28,1)

    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=rstate)

    X_t = X_train[:10000]
    X_test = X_train[10000:]

    y_t = y_train[:10000]
    y_test = y_train[10000:]

    return X_t, X_test, y_t, y_test


def getdict(level):
    target_dict = {
        0: 8,
        1: 2,
        2: 1,
        3: 8,
        4: 5,
        5: 1,
        6: 8,
        7: 1,
        8: 9,
        9: 8
    }

    target_dict01 = {
        0: 0,
        1: 1,
        2: 2,
        3: 8,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 9
    }

    target_dict03 = {
        0: 0,
        1: 1,
        2: 2,
        3: 8,
        4: 4,
        5: 5,
        6: 8,
        7: 7,
        8: 8,
        9: 8
    }

    target_dict05 = {
        0: 8,
        1: 1,
        2: 2,
        3: 8,
        4: 4,
        5: 5,
        6: 8,
        7: 1,
        8: 8,
        9: 8
    }

    if level==0.1:
        return target_dict01
    
    if level==0.3:
        return target_dict03

    if level==0.5:
        return target_dict05

    return target_dict