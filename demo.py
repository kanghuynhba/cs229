# import math
# import random

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
# class Customer(object):

#     # instantiate the class
#     def __init__(self, name, balance=0.0):
#         self.name=name
#         self.balance=balance
    
#     # Instance Methods
#     def withdraw(self, amount):
#         if amount>self.balance: 
#             raise RuntimeError('Amount greater than available balance')
#         self.balance-=amount
#         return self.balance

#     def deposit(self, amount):
#         self.balance+=amount
#         return self.balance

# mario=Customer("Mario Srouji", 1000.0)
# mario.withdraw(100.0)
# Customer.withdraw(mario, 100.0)
# print(mario.name, mario.balance)
# # Attributes are mutable
# mario.name="Bob"
# print(mario.name, mario.balance)

# class Car(object):

#     wheels=4

#     def __init__(self, make, model, wheels, miles, year, sold_on):
#         self.make=make
#         self.model=model
#         self.wheels=wheels
#         self.miles=miles
#         self.year=year
#         self.sold_on=sold_on

#     def sale_price(self):
#         if self.sold_on is not None:
#             return 0.0
#         return 5000.0
    
#     def purchase_price(self):
#         if self.sold_on is None:
#             return 0.0
#         return 8000-(.10*self.miles)

#     # Static Methods
#     def make_car_sound():
#         print("VRoooooommm!")

# Introduce an abstraction allows us to combine these 2 Vehicle classes

# class Vehicle(object):

#     base_sale_price=0
#     wheels=0
    
#     def __init__(self, make, model, wheels, miles, year, sold_on):
#         self.make=make
#         self.model=model
#         self.wheels=wheels
#         self.miles=miles
#         self.year=year
#         self.sold_on=sold_on

#     def sale_price(self):
#         if self.sold_on is not None:
#             return 0.0
#         return 5000.0

#     def purchase_price(self):
#         if self.sold_on is None:
#             return 0.0
#         return self.base_sale_price-(.10*self.miles)
    
#     @abstractmethod
#     def vehicle_type(self):
#         pass
    

# class Car(Vehicle):

#     base_sale_price=8000
#     wheels=4

#     def vehicel_type(self):
#         return 'car'

# class Trunk(Vehicle):

#     base_sale_price=10000
#     wheels=4

#     def vehicel_type(self):
#         return 'Trunk'

# mustang=Car('Ford', 'Mustang')
# print(mustang.wheels)
# Car.make_car_sound()

# class Pet(object):

#     def __init__(self, name, species):
#         self.name=name
#         self.species=species

#     def getName(self):
#         return self.name

#     def getSpecises(self):
#         return self.species

# class Dog(Pet):

#     def __init__(self, name, chase_cats):
#         Pet.__init__(self, name, "Dog")
#         self.chase_cats=chase_cats

#     def chaseCats(self):
#         return self.chase_cats

# class Cat(Pet):

#     def __init__(self, name, hates_dogs):
#         Pet.__init__(self, name, "Cat")
#         self.hates_dogs=hates_dogs

#     def chaseCats(self):
#         return self.hates_dogs

# Numpy

import numpy as np

# Array/matrix initialization

# a=np.array([1, 2, 3])
# print(a)

# b=np.array([[1, 2, 3], [4, 5, 6]])
# print(b)

# c0=np.zeros(5)
# c1=np.ones((5, 5))
# print(c0)
# print(c1)

# d=np.random.random((5, 5))
# print(d)
# print(" Access an element")
# print(d[1,2])
# print(" Access a column")
# print(d[:,2])
# print(" Access a range of columns")
# print(d[:, 0:2])
# print(" Access a range of columns and rows")
# print(d[1:4, 0:2])
# print(" Access elements>2")
# h=d*10
# print(h[h>2])

# e=np.eye(5) # Identity matrix
# print(e)

# Math over Array

# x=np.array([[1, 2], [3, 4]])
# y=np.array([[5, 6], [7, 8]])
# # Add oprt
# print(x+y)
# print(np.add(x, y))
# # Multiplication oprt
# print(x.dot(y))
# print(np.dot(x, y))

# Broadcasting
# We will add the vector v to "each" row of the matrix x,
# storing the result in the matrix y
x=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v=np.array([1, 0, 1])
y=np.empty_like(x) # Create an empty matrix with the same shape as x
# Add the vector x to each row of the matrix x with an explicit loop
for i in range(4):
    y[i:]=x[i:]+v
print(y)