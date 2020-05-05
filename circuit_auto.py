import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def fun_rosenbrock(x):
    return np.sum(100*(x[1:]-x[:-1]**2)**2+(1-x[:-1])**2)

def grad_fun_rosenbrock(x):
    x_j = x[1:-1]
    x_j_m_1 = x[:-2]
    x_j_p_1 = x[2:]
    res = np.zeros_like(x)
    res[1:-1] = -400*x_j*(x_j_p_1 - x_j**2) + 200*(x_j-x_j_m_1**2) - 2*(1-x_j)
    res[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    res[-1] = 200*(x[-1]-x[-2]**2)
    return res

def c(x):
    return x[0]**2+2*x[1]**3-1

def grad_c(x):
    return np.array([2*x[0],6*x[1]**2])

def laplacien(fun,c, lam):
    return lambda x: fun(x)+lam*c(x)

#def uzawa_fixed_step(fun, grad_fun, c, grad_c, x0, l, rho, lambda0 = 1.0, max_iter = 100000, epsilon_grad_L = 1e-8):
#    xk=x0
#    lambda_k=lambda0
#    k=0
#    grad_l_k=grad_fun(xk)+lambda_k*grad_c(xk)
#    pk=0
#    while ((k<max_iter) and (np.linalg.norm(grad_l_k)>epsilon_grad_L)):
#        grad_l_k=grad_fun(xk)+lambda_k*grad_c(xk)
#        pk=-grad_l_k
#        xk+= l*pk
#        lambda_k=max([0,lambda_k+ rho*c(xk)])
#        k+=1
#    print("Nombre d'iterations : ", k)
#    return xk
#
#x0=np.array([0.,0.])
#
#X=np.array([k*0.0001 for k in range(1,101,5)])
#Y=[]
#for a in X:
#    Y=Y+[np.linalg.norm(uzawa_fixed_step(fun_rosenbrock,grad_fun_rosenbrock,c,grad_c, np.array([0., 0.]),a,1)-np.array([0.7684, 0.5894]))]
#Y=np.array(Y)
#plt.plot(X,Y)
#plt.show()
#
#
def cond1(fun, xk,li,pk,c1,grad_fun):
    return (fun(xk+li*pk)<=fun(xk)+c1*li*np.dot(np.transpose(grad_fun(xk)),pk))

def cond2(fun, xk,li,pk,c2,grad_fun):
    return (np.dot(np.transpose(grad_fun(xk+li*pk)),pk) >= c2*np.dot(np.transpose(grad_fun(xk)),pk))

def wolfe_step(fun, grad_fun, xk, pk, c1=0.25, c2=0.75, M=1000):
   l_moins, l_plus = 0, 0
   f_xk = fun(xk)
   grad_f_xk = grad_fun(xk)
   li = 1
   for i in range(M):
      if fun(xk + li * pk) > (f_xk + c1 * li * np.dot(grad_f_xk,pk)):
         l_plus = li
         li = (l_moins + l_plus) / 2.0
      else:
         if np.dot(grad_fun(xk + li * pk), pk) < c2 * np.dot(grad_f_xk, pk):
            l_moins = li
            if l_plus == 0:
               li = 2 * li
            else:
               li = (l_moins + l_plus) / 2.0
         else:
            return li
   return li



#def uzawa_wolfe_step(fun, grad_fun, c, grad_c, x0, rho, lambda0 = 1.0, max_iter = 100000, epsilon_grad_L = 1e-8):
#    xk=x0
#    lambda_k=lambda0
#    k=0
#    grad_l_k=grad_fun(xk)+lambda_k*grad_c(xk)
#    pk=0
#    while ((k<max_iter) and (np.linalg.norm(grad_l_k)>epsilon_grad_L)):
#        grad_l_k=grad_fun(xk)+lambda_k*grad_c(xk)
#        pk=-grad_l_k
#        xk+= wolfe_step(laplacien(fun,c,lambda_k),laplacien(grad_fun,grad_c,lambda_k),xk,pk)*pk
#        lambda_k=max([0,lambda_k+ rho*c(xk)])
#        k+=1
#    print("Nombre d'iterations : ", k)
#    return xk
#
#X=np.array([k*0.0001 for k in range(1,21)])
#Y=[]
#for a in X:
#    Y=Y+[np.linalg.norm(uzawa_wolfe_step(fun_rosenbrock,grad_fun_rosenbrock,c,grad_c, np.array([0., 0.]),a,1)-np.array([0.7684, 0.5894]))]
#Y=np.array(Y)
#plt.plot(X,Y)
#plt.show()
#

def newton_BFGS(fun, grad_fun,x0 , lambda0 = 1.0, max_iter = 100000, epsilon_grad_L = 1e-8):
    k = 0
    xk = x0.copy()
    grad_f_xk = grad_fun(xk)
    Hk = np.identity(len(x0))
    lambda_k=lambda0
    while ((k<max_iter) and (np.linalg.norm(grad_f_xk)>epsilon_grad_L)):
        grad_l_k=grad_f_xk+ lambda_k
        pk = -np.matmul(Hk,grad_f_xk)
        lk = wolfe_step(fun, grad_fun, xk, pk)
        xk1 = xk + lk*pk
        grad_f_xk1 = grad_fun(xk1)
        sk = xk1 - xk
        yk = grad_f_xk1 - grad_f_xk
        gammak = 1.0/np.dot(yk, sk)
        Ak = np.identity(len(x0)) - gammak*np.multiply(sk[:, np.newaxis], yk)
        Bk = np.identity(len(x0)) - gammak*np.multiply(yk[:, np.newaxis], sk)
        Hk = np.matmul(np.matmul(Ak, Hk), Bk) + gammak*np.multiply(sk[:, np.newaxis], sk)
        xk = xk1
        grad_f_xk = grad_f_xk1
        k = k + 1
    print("Nombre d'iterations : ", k)
    return xk


n =  10
#sigma has to be a numpy array
def fun_u(sigma,h0,h1,h2,l,gamma):
    n = len(sigma)
    s = sigma
    res = []
    res.append(1/h1*(l*s[0]*(s[0] - s[n-1]) + h2*l**2*s[0]**2 + h0 + gamma[0]))
    for i in range(1,n):
        res.append(1/h1*(l*s[i]*(s[i] - s[i-1]) + h2*l**2*s[i]**2 + h0 + gamma[i]))
    return res

def fun_v(sigma,l):
    return l*sigma

def fun_P(sigma,b1,b2,h0,h1,h2,l,gamma):
    u = fun_u(sigma,h0,h1,h2,l,gamma)
    v = fun_v(sigma,l)
    return b1*np.multiply(u,v) + b2*np.power(u,2)

def fun_J(sigma,b1=1000,b2=1000,h0=1,h1=1,h2=0.001,l=L/,gamma):
    n = len(sigma)
    J_pondere = np.multiply(np.reciprocal(sigma),J)
    J_pondere[0] /= 2
    J_pondere[n-1] /= 2
    return np.sum(J_pondere)

def grad_fun_J(sigma,b1=1000,b2=1000,h1,h2,l):
    n = len(sigma)
    u = fun_u(sigma,h0,h1,h2,l,gamma)
    v = fun_v(sigma,l)
    res = []
    for i in range(n):
        d_ui_sigmai = 1/h1*(2*l*sigma[i]*(1+h2*l) + l*sigma[i-1])
        dui1_sugmai = -l/h1*sigma[i+1]
        #represente du_(i+1)/dsigma_i
        res.append((b1*l)*(d_ui_sigmai + d_ui1_sigmai) + 2*b2*(u[i+1]/sigma[i+1]*d_ui1_sigmai + u[i]/sigma[i]*d_ui_sigmai) + b2*(u[i]/sigma[i])**2)
    return np.array(res)

    
print(newton_BFGS(fun_rosenbrock,grad_fun_rosenbrock,np.array([0.,0.])))
print(newton_BFGS(fun_J,grad_fun_J,np.array([0.,0.])))
