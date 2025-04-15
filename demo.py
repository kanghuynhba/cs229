# import math
# import random
# random.seed(0)

def sigmoid(x):
    return 1/(1+math.exp(-x))

def func2(*args, **kwargs):
    print(args)
    print(kwargs)

def func1(v, *args, **kwargs):

    func2(args, kwargs)

    if 'power' in kwargs:
        return v**kwargs['power']
    else:
        return v

print(func1(10, 'extra 1', 'extra 2', power=3))
print('--------')
print(func1(10, 5))