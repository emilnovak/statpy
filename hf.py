from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy import linalg as la

matplotlib.rc('figure', figsize=(7, 7))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim3d(0, 8)
ax.set_ylim3d(0, 8)
ax.set_zlim3d(0, 8)

def normalize(vec):
    x = vec[0]
    y = vec[1]
    z = vec[2]
    return (x, y, z) / np.sqrt(x*x + y*y + z*z)


#nagyítási faktor
factor = 10.0

#távolságok - [m]*10
a_distance = .4 * factor
b_distance = .3 * factor
c_distance = .4 * factor

#csúcspontok (x, y, z)
A_point = (a_distance, 0, 0)
B_point = (a_distance, 0, c_distance)
C_point = (a_distance, b_distance, c_distance)
D_point = (0, b_distance, c_distance)
E_point = (0, b_distance, 0)
G_point = (0, 0, 0)

#F_1 vektor - [kN]
F_1_x = 3.0
F_1_y = -1.0
F_1_z = -1.0

#F_2 vektor, látszólag negatív 'y' irányú, a továbbiakban ezzel számoltam - [kN]
F_2_x = 0
F_2_y = -1.0
F_2_z = 0

#F_3 vektor, látszólag pozitív 'x' irányú, a továbbiakban ezzel számoltam - [kN]
F_3_x = 1.3
F_3_y = 0
F_3_z = 0

#M_1 nyomatékvektor - [kNm]
M_1_x = 0.5 
M_1_y = 0.5 
M_1_z = 0.3 

#M_2 nyomatékvektor, látszólag negatív 'z' irányú - [kNm]
M_2_x = 0 
M_2_y = 0
M_2_z = -1.4

#A rúdszerkezet (x, y, z) koordinátáinak listája
x_point_list = (A_point[0], B_point[0], C_point[0], D_point[0], E_point[0])
y_point_list = (A_point[1], B_point[1], C_point[1], D_point[1], E_point[1])
z_point_list = (A_point[2], B_point[2], C_point[2], D_point[2], E_point[2])



#kirajzolás
ax.set_xlabel('X  [10m] [kN] [kNm]')
ax.set_ylabel('Y  [10m] [kN] [kNm]')
ax.set_zlabel('Z  [10m] [kN] [kNm]')

#Pontok elnevezése
ax.text(A_point[0], A_point[1], A_point[2], "A")
ax.text(B_point[0], B_point[1], B_point[2], "B")
ax.text(C_point[0], C_point[1], C_point[2], "C")
ax.text(D_point[0], D_point[1], D_point[2], "D")
ax.text(E_point[0], E_point[1], E_point[2], "E")


#Merev rúdszerkezet
ax.plot(x_point_list, y_point_list, z_point_list, color='black')

#Erővektorok
#F_1
ax.quiver(A_point[0], A_point[1], A_point[2], F_1_x, F_1_y, F_1_z)
ax.text((2*A_point[0]+F_1_x)/2, (2*A_point[1]+F_1_y)/2, (2*A_point[2]+F_1_z)/2, "F1", weight='bold', color='blue')
#F_2
ax.quiver(D_point[0], D_point[1], D_point[2], F_2_x, F_2_y, F_2_z)
ax.text((2*D_point[0]+F_2_x)/2, (2*D_point[1]+F_2_y)/2, (2*D_point[2]+F_2_z)/2, "F2", weight='bold', color='blue')
#F_3
ax.quiver(C_point[0], C_point[1], C_point[2], F_3_x, F_3_y, F_3_z)
ax.text((2*C_point[0]+F_3_x)/2, (2*C_point[1]+F_3_y)/2, (2*C_point[2]+F_3_z)/2, "F3", weight='bold', color='blue')

#Nyomatékvektorok
#M_1
ax.quiver(E_point[0], E_point[1], E_point[2], M_1_x, M_1_y, M_1_z, color='red')
ax.text((2*E_point[0]+M_1_x)/2, (2*E_point[1]+M_1_y)/2, (2*E_point[2]+M_1_z)/2, "M1", weight='bold', color='red')
#M_2
ax.quiver(B_point[0], B_point[1], B_point[2] + abs(M_2_z), M_2_x, M_2_y, M_2_z, color='red')
ax.text((2*B_point[0]+M_2_x)/2, (2*B_point[1]+M_2_y)/2, (2*B_point[2]+abs(M_2_z))/2, "M2", weight='bold', color='red')


#'x', 'y', 'z' tengelyek
ax.quiver(0, 0, 0, 1, 0, 0, color="BLACK")
ax.text(1, 0, 0, "x")
ax.quiver(0, 0, 0, 0, 1, 0, color="BLACK")
ax.text(0, 1, 0, "y")
ax.quiver(0, 0, 0, 0, 0, 1, color="BLACK")
ax.text(0, 0, 1, "z")





#számítások

#origóba redukált vektorkettős
#F_0 origóba redukált erővektor
F_0 = np.array([0, 0, 0])
#M_0 origóba redukált vektorkettős
M_0 = np.array([0, 0, 0])

F_1 = np.array([F_1_x, F_1_y, F_1_z])
r_F_1 = np.array(A_point)

F_2 = np.array([F_2_x, F_2_y, F_2_z])
r_F_2 = np.array(D_point)

F_3 = np.array([F_3_x, F_3_y, F_3_z])
r_F_3 = np.array(C_point)

M_1 = np.array([M_1_x, M_1_y, M_1_z])

M_2 = np.array([M_2_x, M_2_y, M_2_z])

print("F1: " + str(F_1) + " [kN]")
print("rF1: " + str(r_F_1) + " [m]")
print("F2: " + str(F_2) + " [kN]")
print("rF2: " + str(r_F_2) + " [m]")
print("F3: " + str(F_3) + " [kN]")
print("rF3: " + str(r_F_3) + " [m]")
print("M1: " + str(M_1) + " [kNm]")
print("M2: " + str(M_2) + " [kNm]")
print("---------------------------")

F_0 = F_1 + F_2 + F_3



ax.quiver(0, 0, 0, F_0[0], F_0[1], F_0[2], color='blue')
ax.text(F_0[0]/2, F_0[1]/2, F_0[2]/2, "F0", color='blue', weight='bold')
print("F0: " + str(F_0) + " [kN]")

M_0 = M_1 + M_2 + np.cross(r_F_1, F_1) + np.cross(r_F_2, F_2) + np.cross(r_F_3, F_3)
ax.quiver(0, 0, 0, M_0[0], M_0[1], M_0[2], color='red')
ax.text(M_0[0]/2, M_0[1]/2, M_0[2]/2, "M0", color='red', weight='bold')
print("M0: " + str(M_0) + " [kNm]")


#Erő F tengelyre
#Legyen f_vec az F_0 erő egységvektora, m_vec pedig egy erre merőleges vektor. M_Pf a tengelyirányú, M_Pm pedig a tengelyre merőleges komponens
f_vec = normalize(F_0)
m_vec = normalize(np.array([1, 1, (f_vec[0] + f_vec[1])/-f_vec[2]]))
print("f: " + str(f_vec) + " [m]")
print("m: " + str(m_vec) + " [m]")

M_Pf = np.dot(M_0, f_vec)
print("M_Pf: " + str(M_Pf) + " [kNm]")

M_Pm = np.dot(M_0, m_vec)
print("M_Pm: " + str(M_Pm) + " [kNm]")


#centrális egyenes
cent = np.cross(F_0, M_0) / (F_0[0]*F_0[0] + F_0[1]*F_0[1] + F_0[2]*F_0[2])
ax.plot((cent[0]*-10, cent[0]*10), (cent[1]*-10, cent[1]*10), (cent[2]*-10, cent[2]*10), color='green', label='centrális egyenes')
ax.text(cent[0], cent[1], cent[2], "cent", color='green', weight='bold')
print("central line: " + str(cent) + " [m]")

r_G_O = cent * -1

M_G = M_0 + np.cross(r_G_O, F_0)
ax.quiver(0, 0, 0, M_G[0], M_G[1], M_G[2], color='red')
ax.text(M_G[0]/2, M_G[1]/2, M_G[2]/2, "MG", color='red', weight='bold')
print("M_G: " + str(M_G) + " [kNm]")

print("F_0 x M_G: " + str(np.cross(F_0, M_G)))


plt.show()

