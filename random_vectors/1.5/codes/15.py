import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#line.funcs.py
def dir_vec(A,B):
  return B-A

def norm_vec(A,B):
  return np.matmul(omat, dir_vec(A,B))

#Generate line points
def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB

def line_dir_pt(m,A,k1,k2):
  len = 10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(k1,k2,len)
  for i in range(len):
    temp1 = A + lam_1[i]*m
    x_AB[:,i]= temp1.T
  return x_AB


#Intersection of two lines
def line_intersect(n1,A1,n2,A2):
  N=np.vstack((n1,n2))
  p = np.zeros(2)
  p[0] = n1@A1
  p[1] = n2@A2
  #Intersection
  P=np.linalg.inv(N)@p
  return P

#Intersection of two lines
def perp_foot(n,cn,P):
  m = omat@n
  N=np.block([[n],[m]])
  p = np.zeros(2)
  p[0] = cn
  p[1] = m@P
  #Intersection
  x_0=np.linalg.inv(N)@p
  return x_0




#conics.funcs.py
#Generating points on a circle
def circ_gen(O,r):
	len = 50
	theta = np.linspace(0,2*np.pi,len)
	x_circ = np.zeros((2,len))
	x_circ[0,:] = r*np.cos(theta)
	x_circ[1,:] = r*np.sin(theta)
	x_circ = (x_circ.T + O).T
	return x_circ

#Generating points on an ellipse
def ellipse_gen(a,b):
	len = 50
	theta = np.linspace(0,2*np.pi,len)
	x_ellipse = np.zeros((2,len))
	x_ellipse[0,:] = a*np.cos(theta)
	x_ellipse[1,:] = b*np.sin(theta)
	return x_ellipse

#Generating points on a parabola
def parab_gen(y,a):
	x = y**2/a
	return x

#Generating points on a standard hyperbola 
def hyper_gen(y):
	x = np.sqrt(1+y**2)
	return x




#triangle.funcs.py
#Triangle vertices
def tri_vert(a,b,c):
  p = (a**2 + c**2-b**2 )/(2*a)
  q = np.sqrt(c**2-p**2)
  A = np.array([p,q]) 
  B = np.array([0,0]) 
  C = np.array([a,0]) 
  return  A,B,C



#Foot of the Altitude
def alt_foot(A,B,C):
  m = B-C
  n = np.matmul(omat,m) 
  N=np.vstack((m,n))
  p = np.zeros(2)
  p[0] = m@A 
  p[1] = n@B
  #Intersection
  P=np.linalg.inv(N.T)@p
  return P


#Radius and centre of the circumcircle
#of triangle ABC
def ccircle(A,B,C):
  p = np.zeros(2)
  n1 = dir_vec(B,A)
  p[0] = 0.5*(np.linalg.norm(A)**2-np.linalg.norm(B)**2)
  n2 = dir_vec(C,B)
  p[1] = 0.5*(np.linalg.norm(B)**2-np.linalg.norm(C)**2)
  #Intersection
  N=np.vstack((n1,n2))
  O=np.linalg.inv(N)@p
  r = np.linalg.norm(A -O)
  return O,r

#Radius and centre of the incircle
#of triangle ABC
def icircle(A,B,C):
  k1 = 1
  k2 = 1
  p = np.zeros(2)
  t = norm_vec(B,C)
  n1 = t/np.linalg.norm(t)
  t = norm_vec(C,A)
  n2 = t/np.linalg.norm(t)
  t = norm_vec(A,B)
  n3 = t/np.linalg.norm(t)
  p[0] = n1@B- k1*n2@C
  p[1] = n2@C- k2*n3@A
  N=np.vstack((n1-k1*n2,n2-k2*n3))
  I=np.matmul(np.linalg.inv(N),p)
  r = n1@(I-B)
  #Intersection
  return I,r

omat = np.array([[0, 1], [-1, 0]])

#random vertices generated
A=np.array([-6,-3])
B=np.array([-5,-2])
C=np.array([5,-6])

t1 = norm_vec(B,C) 
n1 = t1/np.linalg.norm(t1) 
t2 = norm_vec(C,A)
n2 = t2/np.linalg.norm(t2)
t3 = norm_vec(A,B)
n3 = t3/np.linalg.norm(t3)

#parameters
n_a=dir_vec(n3,n2)
n_b=dir_vec(n1,n3)
n_c=dir_vec(n1,n2)

m_a=norm_vec(n2,n3)
m_b=norm_vec(n1,n3)
m_c=norm_vec(n1,n2)
c_a=n_a@A
c_b=n_b@B
c_c=n_c@C

print(f"m,n,c of bistor of A : {m_a},{n_a},{c_a}")
print(f"m,n,c of bistor of B : {m_b},{n_b},{c_b}")
print(f"m,n,c of bistor of C : {m_c},{n_c},{c_c}")

I,r=icircle(A,B,C)

def angle_btw_vectors(v1, v2):
    dot_product = v1 @ v2
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = np.arccos(dot_product / norm)
    angle_in_deg = np.degrees(angle)
    return angle_in_deg
    
#Calculating the angles split by bisectors
angle_BAI = angle_btw_vectors(dir_vec(B,A), dir_vec(I,A))
angle_CAI = angle_btw_vectors(dir_vec(C,A), dir_vec(I,A))
print("Angle BAI:", angle_BAI)
print("Angle CAI:", angle_CAI)
angle_ABI = angle_btw_vectors(dir_vec(A,B), dir_vec(I,B))
angle_CBI = angle_btw_vectors(dir_vec(C,B), dir_vec(I,B))
print("Angle ABI:", angle_ABI)
print("Angle CBI:", angle_CBI)
angle_ACI = angle_btw_vectors(dir_vec(A,C), dir_vec(I,C))
angle_BCI = angle_btw_vectors(dir_vec(B,C), dir_vec(I,C))
print("Angle ACI:", angle_ACI)
print("Angle BCI:", angle_BCI)

r1 = abs((t3@I) - (t3@A))/(np.linalg.norm(t3))   #r1 is distance between I and AB
r2 = abs((t2@I) - (t2@C))/(np.linalg.norm(t2))   #r2 is distance between I and AC
r3 = abs((t1@I) - (t1@C))/(np.linalg.norm(t1))   #r2 is distance between I and BC

print("Distance between I and AB is",r1)
print("Distance between I and BC is",r2)
print("Distance between I and AC is",r3)

#finding k for D_3,E_3 and F_3
k2=((I-A)@(A-B))/((A-B)@(A-B))
k1=((I-A)@(A-C))/((A-C)@(A-C))
k3=((I-B)@(B-C))/((B-C)@(B-C))
#finding D_3,E_3 and F_3
E3=A+(k1*(A-C))
F3=A+(k2*(A-B))
D3=B+(k3*(B-C))
print("E3 = ",E3)
print("F3 = ",F3)
print("D3 = ",D3)
incircle=circ_gen(I,r)
print(f"I={I},r={r}")
#plot
plt.plot(incircle[0,:],incircle[1,:],label='$incircle$')
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_AI = line_gen(A,I)
x_BI = line_gen(B,I)
x_CI = line_gen(C,I)
#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_AI[0,:],x_AI[1,:],label='$AI$')
plt.plot(x_BI[0,:],x_BI[1,:],label='$BI$')
plt.plot(x_CI[0,:],x_CI[1,:],label='$CI$')

A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
D3 = D3.reshape(-1,1)
E3 = E3.reshape(-1,1)
F3 = F3.reshape(-1,1)
I = I.reshape(-1,1)
tri_coords = np.block([[A,B,C,D3,E3,F3,I]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','$D_3$','$E_3$','$F_3$','I']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(-10,0), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

plt.show()
