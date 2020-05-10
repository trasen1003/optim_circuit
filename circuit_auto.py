import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import numdifftools as nd
import math as math
import pandas as pd

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

def wolfe_step(fun, grad_fun, xk, pk, lambdaa, c1=0.25, c2=0.75, M=100):
   l_moins, l_plus = 0, math.inf
   f_xk = fun(xk, lambdaa)
   grad_f_xk = grad_fun(xk, lambdaa)
   li = 1
   for i in range(M):
      print(li)
      if fun(xk + li * pk , lambdaa) > (f_xk + c1 * li * np.dot(grad_f_xk,pk)):
         l_plus = li
         li = (l_moins + l_plus) / 2.0
      else:
         if np.dot(grad_fun(xk + li * pk , lambdaa), pk) < c2 * np.dot(grad_f_xk, pk):
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

def newton_BFGS(fun, grad_fun,x0 , lambdaa =[0], max_iter = 1000, epsilon_grad_L = 1e-8):
    k = 0
    xk = x0.copy()
    grad_f_xk = grad_fun(xk,lambdaa)
    Hk = np.identity(len(x0))
    lambda_k=lambdaa
    while ((k<max_iter) and (np.linalg.norm(grad_f_xk)>epsilon_grad_L)):
        pk = -np.matmul(Hk,grad_f_xk)
        lk = wolfe_step(fun, grad_fun, xk, pk,lambdaa)
        xk1 = xk + lk*pk
        grad_f_xk1 = grad_fun(xk1,lambdaa)
        sk = xk1 - xk
        yk = grad_f_xk1 - grad_f_xk
        gammak = 1.0/np.dot(yk, sk)
        Ak = np.identity(len(x0)) - gammak*np.multiply(sk[:, np.newaxis], yk)
        Bk = np.identity(len(x0)) - gammak*np.multiply(yk[:, np.newaxis], sk)
        Hk = np.matmul(np.matmul(Ak, Hk), Bk) + gammak*np.multiply(sk[:, np.newaxis], sk)
        xk = xk1
        grad_f_xk = grad_f_xk1
        print(k)
        k = k + 1
    print("Nombre d'iterations : ", k)
    return xk


#La contrainte égalité du problème impose sigma0=sigma_n-1 et on peut donc enlever une dimension au problème. Ainsi on considérera que sigma est de dimension n-1, on se retrouve alors avec des contraintes inégalités que l'on sait résoudre.




gamma_test=np.array([0,0,0])





#sigma has to be a numpy array
def fun_u(sigma,h0,h1,h2,gamma,l=1):
    n = len(sigma)
    s = sigma
    res = []
    res.append(1/h1*(h2*l**2*(s[0])**2 + h0 + gamma[0]))
    for i in range(1,n):
        res.append(1/h1*(l*s[i]*(s[i] - s[i-1]) + h2*l**2*s[i]**2 + h0 + gamma[i]))
    res.append(1/h1*(l*s[0]*(s[0]-s[n-1]) + h2*l**2*s[0]**2+h0+gamma[0]))
    return np.array(res)


def fun_v(sigma,l):
    return l*sigma

def fun_P(sigma,b1,b2,h0,h1,h2,gamma,l=1):
    u = fun_u(sigma,h0,h1,h2,gamma,l)
    v = np.concatenate((fun_v(sigma,l),np.array([l*sigma[0]])))
    return b1*np.multiply(u,v) + b2*np.power(u,2)

def fun_J(sigma,gamma=gamma_test, b1=1000,b2=1000,h0=1,h1=1,h2=0.001,l=1):
    n = len(sigma)
    sigma2=np.concatenate((sigma,[sigma[0]]))
    J_pondere = np.multiply(np.reciprocal(sigma2),fun_P(sigma,b1,b2,h0,h1,h2,gamma,l))
    J_pondere[0] /= 2
    J_pondere[n-1] /= 2
    return np.sum(J_pondere)

def grad_fun_J(sigma,gamma=gamma_test,b1=1000,b2=1000,h0=1,h1=1,h2=0.001,l=1):
    n = len(sigma)
    u = fun_u(sigma,h0,h1,h2,gamma,l)
    v = fun_v(sigma,l)
    res = []
    d_ui_sigmai = 2/h1*l*sigma[0]*(h2*l)
    d_ui1_sigmai = -l/h1*sigma[1]
    res.append((b1*l)*(d_ui_sigmai + d_ui1_sigmai) + 2*b2*(u[1]/sigma[1]*d_ui1_sigmai + u[0]/sigma[0]*d_ui_sigmai) - b2*(u[0]/sigma[0])**2)
    for i in range(1,n-1):
        d_ui_sigmai = 1/h1*(2*l*sigma[i]*(1+h2*l) + l*sigma[i-1])
        d_ui1_sigmai = -l/h1*sigma[i+1]
        #represente du_(i+1)/dsigma_i
        res.append((b1*l)*(d_ui_sigmai + d_ui1_sigmai) + 2*b2*(u[i+1]/sigma[i+1]*d_ui1_sigmai + u[i]/sigma[i]*d_ui_sigmai) - b2*(u[i]/sigma[i])**2)
    d_ui_sigmai = 1/h1*(2*l*sigma[n-1]*(1+h2*l) + l*sigma[n-2])
    d_ui1_sigmai = -l/h1*sigma[0]
    res.append((b1*l)*(d_ui_sigmai + d_ui1_sigmai) + 2*b2*(u[0]/sigma[0]*d_ui1_sigmai + u[n-1]/sigma[n-1]*d_ui_sigmai) - b2*(u[n-1]/sigma[n-1])**2)

    return np.array(res)

def grad_fun_J_df(sigma,gamma=gamma_test,h=0.001,b1=1000,b2=1000,h0=1,h1=1,h2=0.001,l=1):
    res=[]
    tab=sigma.copy()
    n=len(sigma)
    tab[0] +=h
    for i in range(n-1):
        res.append((fun_J(tab,gamma)-fun_J(sigma,gamma))/h)
        tab[i] -=h
        tab[i+1] +=h
    res.append((fun_J(tab,gamma)-fun_J(sigma,gamma))/h)
    return res


test=nd.Hessian(fun_J)

vmin= 1
vmax=300
def c1(sigma):
    n=len(sigma)
    Vmin=np.array(n*[vmin])
    V=l*sigma
    tab= Vmin-V
    return tab


def c2(sigma):
    n=len(sigma)
    Vmax=np.array(n*[vmin])
    V=l*sigma
    tab= V-Vmax
    return tab

def c3(sigma,gamma,b1=1000,b2=1000,h0=1,h1=1,h2=0.001,l=1):
    return -fun_u(sigma,h0,h1,h2,gamma)

def c4(sigma,gamma,b1=1000,b2=1000,h0=1,h1=1,h2=0.001,l=1):
    n=len(sigma)
    return fun_u(sigma,h0,h1,h2,gamma)-np.ones(n)

def c5(sigma):
    return sigma[-1]-sigma[0]

def A(i,n,b1=1000,b2=1000,h0=1,h1=1,h2=0.001,l=1):
    res=np.zeros([n,n])
    if (i !=0 and i != (n)):
        res[i,i]=2/h1*(l+h2*l**2)
        res[i,i-1]=-l/h1
        res[i-1,i]=-l/h1
    elif i==0:
        res[0,0]=2*h2/h1*l**2
    else:
        res[0,0]=1/(2*h1)*(l+h2*l**2)
        res[0,n-1]=-l/h1
        res[n-1,0]=-l/h1
    return res

def Lagrangien(sigma,lambdaa,gamma=gamma_test,b1=1000,b2=1000,h0=1,h1=1,h2=0.001,l=1):
    n=len(sigma)
    m=n+1
    lambda1=lambdaa[0:n]
    lambda2=lambdaa[n:2*n]
    lambda3=lambdaa[2*n:3*n+1]
    lambda4=lambdaa[3*n+1:]
    res=0
    for i in range(n):
        Ai=A(i,n,b1,b2,h0,h1,h2,l)
        res+= (lambda1[i]-lambda2[i])+l*sigma[i]
        res+=0.5*(-lambda3[i]+lambda4[i])*(np.dot(np.dot(sigma,Ai),np.transpose(sigma)))+(h0+gamma[i])/h1
    res+=0.5*(-lambda3[n]+lambda4[n])*(np.dot(np.dot(sigma,Ai),np.transpose(sigma))+(h0+gamma[i])/h1)
    return res+ fun_J(sigma,gamma,b1,b2,h0,h1,h2,l)



def grad_Lagrangien(sigma,lambdaa,h=0.0001,b1=1000,b2=1000,h0=1,h1=1,h2=0.001,l=1):
    n=len(sigma)
    m=n+1
    lambda1=lambdaa[0:n]
    lambda2=lambdaa[n:2*n]
    lambda3=lambdaa[2*n:3*n+1]
    lambda4=lambdaa[3*n+1:]
    res=np.zeros((n))
    a=np.sum(lambda1)
    b=np.sum(lambda2)
    var1=np.zeros((n))
    var2=np.zeros((n))
    res+= (a-b)*np.ones((n))*l
    for i in range(n+1):
        Ai=A(i,n,b1,b2,h0,h1,h2,l)
        res+= (-lambda3[i]+lambda4[i])*np.dot(Ai,np.transpose(sigma))

    return res+grad_fun_J_df(sigma,gamma_test,h,b1,b2,h0,h1,h2,l)

print(newton_BFGS(Lagrangien,grad_Lagrangien,np.array([1.,3.]),np.array([1,1,1,1,1,1,1,1,1,1])))

file=pd.read_csv('trajectoire_reference.csv')


