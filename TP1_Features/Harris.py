import numpy as np
import cv2

from matplotlib import pyplot as plt

#Lecture image en niveau de gris et conversion en float64
img=np.float64(cv2.imread('Image_Pairs/Graffiti0.png',cv2.IMREAD_GRAYSCALE))
print(np.shape(img))
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")
print("Type de l'image :",img.dtype)

#Début du calcul
t1 = cv2.getTickCount()
Theta = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
print("Shape de la matrice de autocorrelation: ",np.shape(Theta))
# Mettre ici le calcul de la fonction d'intérêt de Harris

#----------

#we have to make sure all results are in the [0,255] range, so
def adjust_values(f_result):

    f_result_norm = 255*(f_result - f_result.min()) / (f_result.max() - f_result.min())
    return f_result_norm

#On commence en calculant la matrice de autocorrelation


def calc_auto(in_matrix,shape,alpha):
    len_window = shape[1]
    size=len(in_matrix)
    size_y=np.shape(in_matrix)[1]
    f_result=np.zeros((size,size_y))
    kernel_dx=np.array([[0,0,0],[-1,0,1],[0,0,0]])

    kernel_dy=np.array([[0,-1,0],[0,0,0],[0,1,0]])

    matrix_dx=cv2.filter2D(in_matrix,-1,kernel_dx)
    matrix_dx = adjust_values(matrix_dx)
    matrix_dy=cv2.filter2D(in_matrix,-1,kernel_dy)
    matrix_dy=adjust_values(matrix_dy)
    
    #we also pad the original image's derivatives with 0s to make the loop
    matrix_copy_dx = cv2.copyMakeBorder(matrix_dx, 1, 1, 1, 1,cv2.BORDER_CONSTANT)
    matrix_copy_dy = cv2.copyMakeBorder(matrix_dy, 1, 1, 1, 1,cv2.BORDER_CONSTANT)
    #onc commence le boucle:
    for i in range(1,size + 1):
        for j in range(1,size_y+1):
            #on cree deux matrices pour calculer l autocorrelation
            window_dx = np.array([[matrix_copy_dx[k][p] for p in range(j-1,j+2)] for k in range(i-1,i+2)])#wrong
            window_dy = np.array([[matrix_copy_dy[k][p] for p in range(j-1,j+2)] for k in range(i-1,i+2)])
            auto_corr = np.zeros((2,2))#la matrice qu'on sauvegardera ls valeurs
            auto_corr[0][0]=np.sum(window_dx**2)
            auto_corr[1][1]=np.sum(window_dy**2)
            auto_corr[0][1]=np.sum(np.multiply(window_dx,window_dy))
            auto_corr[1][0]=np.sum(np.multiply(window_dx,window_dy))
            f_result[i-1][j-1] =np.linalg.det(auto_corr) - alpha*(np.trace(auto_corr)**2)
    

    return f_result


Theta=calc_auto(Theta,(3,3),0.05)
#Theta = adjust_values(Theta)
print('Shape of Theta after operation: ',np.shape(Theta))
# neg_values = np.sum(np.int16(Theta < 0))
# high_values = np.sum(np.int16(Theta>255))
# print('AMount of wrong values(> 255 or <0) in Theta', neg_values+high_values)
# Calcul des maxima locaux et seuillage
Theta_maxloc = cv2.copyMakeBorder(Theta,0,0,0,0,cv2.BORDER_REPLICATE)
d_maxloc = 3
seuil_relatif = 0.01
se = np.ones((d_maxloc,d_maxloc),np.uint8)
Theta_dil = cv2.dilate(Theta,se)
#Theta_dil = adjust_values(Theta_dil)
#Suppression des non-maxima-locaux
Theta_maxloc[Theta < Theta_dil] = 0.0
#On néglige également les valeurs trop faibles
Theta_maxloc[Theta < seuil_relatif*Theta.max()] = 0.0
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Mon calcul des points de Harris :",time,"s")
print("Nombre de cycles par pixel :",(t2 - t1)/(h*w),"cpp")

plt.subplot(131)
plt.imshow(img,cmap = 'gray')
plt.title('Image originale')

plt.subplot(132)
plt.imshow(Theta,cmap = 'gray')
plt.title('Fonction de Harris')

se_croix = np.uint8([[1, 0, 0, 0, 1],
[0, 1, 0, 1, 0],[0, 0, 1, 0, 0],
[0, 1, 0, 1, 0],[1, 0, 0, 0, 1]])
print('Shape of max_loc: ',np.shape(Theta_maxloc))
Theta_ml_dil = cv2.dilate(Theta_maxloc,se_croix)
#Theta_ml_dil = adjust_values(Theta_ml_dil)
print('Shape of ml_dil: ',np.shape(Theta_ml_dil))
#Relecture image pour affichage couleur
Img_pts=cv2.imread('Image_Pairs/Graffiti0.png',cv2.IMREAD_COLOR)
(h,w,c) = Img_pts.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes x",c,"canaux")
print("Type de l'image :",Img_pts.dtype)
#On affiche les points (croix) en rouge
Img_pts[Theta_ml_dil > 0] = [255,0,0]
plt.subplot(133)
plt.imshow(Img_pts)
plt.title('Points de Harris')

plt.show()
