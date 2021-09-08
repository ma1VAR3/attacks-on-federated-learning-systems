def getmnistclean(rstate):
    import numpy as np
    from keras.datasets import mnist
    from sklearn.model_selection import train_test_split
    from keras.utils.np_utils import to_categorical

    (x_train, y_train), (x_te, y_te) = mnist.load_data()
    x_train=np.reshape(x_train,(x_train.shape[0],-1))
    x_te=np.reshape(x_te,(x_te.shape[0],-1))
    
    x=[]
    y=[]
    for i in range(len(y_train)):
        x.append(x_train[i])
        y.append(y_train[i])
    
    xt=[]
    yt=[]
    for i in range(len(y_te)):
        xt.append(x_te[i])
        yt.append(y_te[i])

    xt=np.array(xt)
    yt=np.array(yt)

    x=np.array(x)
    y=np.array(y)
    
    x = x / 255.0
    # x = x.reshape(-1,28,28,1)

    xt = xt / 255.0
    # xt = xt.reshape(-1,28,28,1)

    y = to_categorical(y)
    yt = to_categorical(yt)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=rstate)

    X_train = X_train[:4000]


    y_train = y_train[:4000]


    return X_train, xt, y_train, yt

def getmnistpoisoned(rstate, level):
    
    import numpy as np
    from keras.datasets import mnist
    from sklearn.model_selection import train_test_split
    from keras.utils.np_utils import to_categorical

    (x_train, y_train), (x_te, y_te) = mnist.load_data()
    x_train=np.reshape(x_train,(x_train.shape[0],-1))

    
    target_dict = {
        0: 5,
        1: 8,
        2: 9,
        3: 7,
        4: 1,
        5: 0,
        6: 4,
        7: 6,
        8: 2,
        9: 3
    }

    target_dict01 = {
        0: 0,
        1: 1,
        2: 2,
        3: 7,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 9
    }

    target_dict03 = {
        0: 5,
        1: 1,
        2: 2,
        3: 7,
        4: 4,
        5: 0,
        6: 6,
        7: 7,
        8: 8,
        9: 9
    }

    target_dict05 = {
        0: 5,
        1: 1,
        2: 9,
        3: 7,
        4: 4,
        5: 0,
        6: 6,
        7: 7,
        8: 2,
        9: 9
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
    # x = x.reshape(-1,28,28,1)

    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=rstate)

    X_train = X_train[:4000]


    y_train = y_train[:4000]


    return X_train, X_test, y_train, y_test


def getdict(level):
    target_dict = {
        0: 5,
        1: 8,
        2: 9,
        3: 7,
        4: 1,
        5: 0,
        6: 3,
        7: 6,
        8: 2,
        9: 3
    }

    target_dict01 = {
        0: 0,
        1: 1,
        2: 2,
        3: 7,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 9
    }

    target_dict03 = {
        0: 5,
        1: 1,
        2: 2,
        3: 7,
        4: 4,
        5: 0,
        6: 6,
        7: 7,
        8: 8,
        9: 9
    }

    target_dict05 = {
        0: 5,
        1: 1,
        2: 9,
        3: 7,
        4: 4,
        5: 0,
        6: 6,
        7: 7,
        8: 2,
        9: 9
    }

    if level==0.1:
        return target_dict01
    
    if level==0.3:
        return target_dict03

    if level==0.5:
        return target_dict05

    return target_dict