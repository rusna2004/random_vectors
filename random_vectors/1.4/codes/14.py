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

# Midpoint of each line
def midpoint(P, Q):
    return (P + Q) / 2 
    
D=midpoint(B,C)
E=midpoint(C,A)
F=midpoint(A,B) 
    
def perp_bisec(B, C):
    midBC=midpoint(B,C)
    n=dir_vec(B,C)
    m=norm_vec(B,C)
    c= n@midBC
    return m,n,c

#parameters of perpendicular bisectors
m_OD,n_OD,c_OD=perp_bisec(B,C)
print(f"m,n,c of OD : {m_OD},{n_OD},{c_OD}")
m_OE,n_OE,c_OE=perp_bisec(A,C)
print(f"m,n,c of OE : {m_OE},{n_OE},{c_OE}")
m_OF,n_OF,c_OF=perp_bisec(B,A)
print(f"m,n,c of OF : {m_OF},{n_OF},{c_OF}")

#point O 
O=line_intersect(n_OE,E,n_OF,F)
print(f"O:{O}")

#distance from vertices 
len_OA=np.linalg.norm(dir_vec(O,A))
len_OB=np.linalg.norm(dir_vec(O,B))
len_OC=np.linalg.norm(dir_vec(O,C))

print(f"the distance from vertices are OA:{len_OA},OB:{len_OB},OC:{len_OC}")

def find_Angle(O,B,C):
    dot_pt_O = (B - O) @ ((C - O).T)
    norm_pt_O = np.linalg.norm(B - O) * np.linalg.norm(C - O)
    cos_theta_O = dot_pt_O / norm_pt_O
    angle_BOC = round(np.degrees(np.arccos(cos_theta_O)),5)  #Round is used to round of number till 5 decimal places
    return angle_BOC

angle_BOC=find_Angle(O,B,C)
angle_BAC=find_Angle(A,B,C)
angle_AOC=find_Angle(O,A,C)
angle_ABC=find_Angle(B,A,C)
angle_AOB=find_Angle(O,A,B)
angle_BCA=find_Angle(C,B,A)
print(f"BOC={angle_BOC},BAC={angle_BAC}")
print(f"AOC={angle_AOC},ABC={angle_ABC}")
print(f"AOB={360-angle_AOB},BCA={angle_BCA}")
[O,r] = ccircle(A,B,C)
x_ccirc= circ_gen(O,r)
#plot
plt.plot(x_ccirc[0,:],x_ccirc[1,:],label='$circumcircle$')
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_OD = line_gen(O,D)
x_OE = line_gen(O,E)
x_OF = line_gen(O,F)
#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_OD[0,:],x_OD[1,:],label='$OD$')
plt.plot(x_OE[0,:],x_OE[1,:],label='$OE$')
plt.plot(x_OF[0,:],x_OF[1,:],label='$OF$')

A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
D = D.reshape(-1,1)
E = E.reshape(-1,1)
F = F.reshape(-1,1)
O = O.reshape(-1,1)
tri_coords = np.block([[A,B,C,D,E,F,O]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D','E','F','O']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

plt.show()
