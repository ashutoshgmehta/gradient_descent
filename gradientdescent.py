import numpy as np

X=[0.5, 2.5]
Y=[0.2, 0.9]

def f(w,b,x):
    return 1.0/(1.0+np.exp(-(w*x+b)))

def error(w,b,x):
    err = 0.0
    for x, y in zip(X,Y):
        fx = f(w,b,x)
        err += 0.5*(fx-y)**2
    return err

def bias(w,b,x,y):
    fx = f(w,b,x)
    return (fx-y)*fx*(1-fx)

def weight(w,b,x,y):
    fx = f(w,b,x)
    return (fx-y)*fx*(1-fx)*x

def gradient():
    w, b, eta, max_epochs = 1,1,0.01,100
    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += weight(w,b,x,y)
            db += bias(w,b,x,y)
        w = w-eta*dw
        b = b-eta*db
        er=error(w, b, x)
    print("Error for noraml gradient descent:\t", er)
