from sympy import *
x1,x2,q1,q2 =var('x1,x2,q1,q2')
X = Matrix([x1,x2])
Q = Matrix([q1,q2])
Cubic_Kernel = (X.dot(Q)+1)*(X.dot(Q)+1)*(X.dot(Q)+1)
print(Cubic_Kernel.expand())

'''


[   x_1^3 ,
    x_2^3,
    3^(1/2)*x_1^2*x_2,
    3^(1/2)*x_1*x_2^2,
    3^(1/2)*x_1^2,
    3^(1/2)*x_2^2,
    6^(1/2)*x_1*x_2,
    3^(1/2)*x_1,
    3^(1/2)*x_2,
    1
    ]

'''
