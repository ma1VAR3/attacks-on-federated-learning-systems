def getmnistclean():
    import numpy as np
    from keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train=np.reshape(x_train,(x_train.shape[0],-1))
    print(x_train.shape)
    
    x_np=[]
    y_np=[]
    k=0
    for i in range(len(y_train)):
        x_np.append(x_train[i])
        y_np.append(y_train[i])
    x_np=np.array(x_np)
    y_np=np.array(y_np)
    x_nptrain=x_np[:10000]
    y_nptrain=y_np[:10000]
    print(x_nptrain.shape,y_nptrain.shape)
    x_nptest=x_np[10000:12000]
    y_nptest=y_np[10000:12000]
    print(x_nptest.shape,y_nptest.shape)

    return x_nptrain, y_nptrain, x_nptest, y_nptest

def getmnistpoisoned(type="targetted", level=0.1):
    
    import numpy as np
    from keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train=np.reshape(x_train,(x_train.shape[0],-1))
    print(x_train.shape)
    
    target_dict = {
        0: 8,
        1: 1,
        2: 1,
        3: 8,
        4: 5,
        5: 1,
        6: 8,
        7: 1,
        8: 8,
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

    x_tp=[]
    y_tp=[]

    # if type=="targetted":
    #     for i in range(len(x_train)):
    #         x_tp.append(x_train[i])
    #         if level > 0.5:
    #             y_tp.append(target_dict[y_train[i]])
    #         elif level == 0.5:
    #             y_tp.append(target_dict05[y_train[i]])
    #         elif level == 0.3:
    #             y_tp.append(target_dict03[y_train[i]])
    #         else:
    #             y_tp.append(target_dict01[y_train[i]])

    


    x_tp=np.array(x_tp)
    y_tp=np.array(y_tp)
    print(x_tp.shape,y_tp.shape)

    x_tptrain=x_tp[:10000]
    y_tptrain=y_tp[:10000]
    print(x_tptrain.shape,y_tptrain.shape)
    x_tptest=x_tp[10000:12000]
    y_tptest=y_tp[10000:12000]
    print(x_tptest.shape,y_tptest.shape)

    return x_tptrain, y_tptrain, x_tptest, y_tptest


def getdict(level):
    target_dict = {
        0: 8,
        1: 1,
        2: 1,
        3: 8,
        4: 5,
        5: 1,
        6: 8,
        7: 1,
        8: 8,
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