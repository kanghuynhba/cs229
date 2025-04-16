import math
import random

# def sigmoid(x):
#     return 1/(1+math.exp(-x))

# def func2(*args, **kwargs):
#     print(args)
#     print(kwargs)

# def func1(v, *args, **kwargs):

#     func2(args, kwargs)

#     if 'power' in kwargs:
#         return v**kwargs['power']
#     else:
#         return v

# Loops
# l=[1, 2, 3]
# for element in l:
#     print(element)


# for i in range(len(l)):
#     print(l[i])

# i=5

# while(i>0):
#     print(i)
#     i-=1

# while(True):
#     print(i)
#     i-=1    
#     if i<=0:
#         break

# Python OOP
class Customer(object):

    # instantiate the class
    def __init__(self, name, balance=0.0):
        self.name=name
        self.balance=balance
    
    # Methods
    def withdraw(self, amount):
        if amount>self.balance: 
            raise RuntimeError('Amount greater than available balance')
        self.balance-=amount
        return self.balance

    def deposit(self, amount):
        self.balance+=amount
        return self.balance

mario=Customer("Mario Srouji", 1000.0)
mario.withdraw(100.0)
Customer.withdraw(mario, 100.0)
print(mario.name, mario.balance)
# Attributes are mutable
mario.name="Bob"
print(mario.name, mario.balance)