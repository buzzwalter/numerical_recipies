import numpy as np

#Jocobi iterative solver
def jocobi_solver(A,x_0, b):
    x_tilde = []
    e = 1.0e-5
    do_while = True
    max = 100000
    counter =  0
    while(do_while):
        M = np.dot(A,x_0)
        x_tilde = np.array([-np.dot(np.delete(A[i],i),np.delete(x_0,i))/A[i][i] + b[i]/A[i][i] for i in range(len(A))])
        if(e > np.linalg.norm(x_tilde - x_0)):
            do_while = False
        x_0 = x_tilde
        counter = counter + 1
        if(counter > max):
            do_while = False
    return np.round(x_tilde,3)

#Gauss-Seidel iterative solver
def gauss_seidel_solver(A,x_0, b):
    x_tilde = x_0
    e = 1.0e-5
    do_while = True
    max = 100000
    counter =  0
    while(do_while):
        M = np.dot(A,x_0)
        x_tilde = np.array([-np.dot(np.delete(A[i],i),np.delete(x_tilde,i))/A[i][i] + b[i]/A[i][i] for i in range(len(A))])
        if(e > np.linalg.norm(x_tilde - x_0)):
            do_while = False
        x_0 = x_tilde
        counter = counter + 1
        if(counter > max):
            do_while = False
    return np.round(x_tilde,3)

# Example using diagonally dominant 3x3
A = np.array([[3, 1, 1], [1, 3, -1], [1,1,-5]])
x_0 = np.array([0,0,0])
b = np.array([5,3,-1])
x_jacobi = jocobi_solver(A,x_0,b)
x_gauss = gauss_seidel_solver(A,x_0,b)
print("Jacobi Solution: ", x_jacobi)
print("Gauss-Seidel Solution: ", x_gauss)