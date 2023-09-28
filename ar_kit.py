import cv2
import sys
import os
import numpy as np
import scipy as sp
import glob
from scipy import signal
from scipy.linalg import solve
from math import dist

def cuboidpyramid(worldpts):
	frame = cv2.imread(worldimg)
	imgpts = []
	xscale=1000
	yscale=1000
	for pt in worldpts:
		wpt = [pt[0],pt[1],pt[2],1]
		wpt = np.array(wpt)
		ipt = np.matmul(P,wpt)
		ipt = ipt/ipt[2]
		iptx = int(ipt[0]*xscale)
		ipty = int(ipt[1]*yscale)
		imgpts.append([iptx,ipty])
		frame = cv2.circle(frame, (iptx, ipty), 15, (0, 0, 255), 15)

	number_of_pts = len(imgpts)

	line_color = (0,0,0)
	# line_color = (0,255,255)


	## Filling with color
	pts1 = np.array([imgpts[0],imgpts[1],imgpts[2],imgpts[3]])
	frame = cv2.fillPoly(frame,pts=[pts1],color = (255,153,51))

	## Filling with color
	pts1 = np.array([imgpts[4],imgpts[5],imgpts[6],imgpts[7]])
	frame = cv2.fillPoly(frame,pts=[pts1],color = (102,255,178))


	pts1 = np.array([imgpts[0],imgpts[1],imgpts[4]])
	# frame = cv2.fillPoly(frame,pts=[pts1],color = (102,255,178))

	pts1 = np.array([imgpts[1],imgpts[2],imgpts[4]])
	# frame = cv2.fillPoly(frame,pts=[pts1],color = (102,255,178))

	pts1 = np.array([imgpts[2],imgpts[3],imgpts[4]])
	# frame = cv2.fillPoly(frame,pts=[pts1],color = (102,255,178))

	pts1 = np.array([imgpts[3],imgpts[0],imgpts[4]])
	# frame = cv2.fillPoly(frame,pts=[pts1],color = (102,255,178))

	## create base (draw lines)
	for i in range(0,4-1):
		frame = cv2.line(frame, imgpts[i], imgpts[i+1], line_color, 15)
	frame = cv2.line(frame, imgpts[0], imgpts[3], line_color, 15)

	## create upperlayer (draw lines)
	for i in range(4,8-1):
		frame = cv2.line(frame, imgpts[i], imgpts[i+1], line_color, 15)
	frame = cv2.line(frame, imgpts[4], imgpts[7], line_color, 15)

	## create sides (draw lines)
	for i in range(0,4):
		frame = cv2.line(frame, imgpts[i], imgpts[i+4], line_color, 15)

	## START OF PYRAMID LAYER
	temppts = worldpts[4:]

	imgpts = []
	xscale=1000
	yscale=1000
	for pt in temppts:
		wpt = [pt[0],pt[1],pt[2],1]
		wpt = np.array(wpt)
		ipt = np.matmul(P,wpt)
		ipt = ipt/ipt[2]
		iptx = int(ipt[0]*xscale)
		ipty = int(ipt[1]*yscale)
		imgpts.append([iptx,ipty])
		frame = cv2.circle(frame, (iptx, ipty), 15, (0, 0, 255), 15)

	number_of_pts = len(imgpts)

	line_color = (0,0,0)
	# line_color = (0,255,255)


	## Filling with color
	pts1 = np.array([imgpts[0],imgpts[1],imgpts[2],imgpts[3]])
	frame = cv2.fillPoly(frame,pts=[pts1],color = (127,0,255))

	pts1 = np.array([imgpts[0],imgpts[1],imgpts[4]])
	# frame = cv2.fillPoly(frame,pts=[pts1],color = (102,255,178))

	pts1 = np.array([imgpts[1],imgpts[2],imgpts[4]])
	frame = cv2.fillPoly(frame,pts=[pts1],color = (102,255,178))

	pts1 = np.array([imgpts[2],imgpts[3],imgpts[4]])
	# frame = cv2.fillPoly(frame,pts=[pts1],color = (102,255,178))

	pts1 = np.array([imgpts[3],imgpts[0],imgpts[4]])
	frame = cv2.fillPoly(frame,pts=[pts1],color = (102,255,178))

	## create base (draw lines)
	for i in range(0,number_of_pts-1):
		frame = cv2.line(frame, imgpts[i], imgpts[i+1], line_color, 15)
	frame = cv2.line(frame, imgpts[0], imgpts[3], line_color, 15)

	## create upper part of pyramid (draw lines)
	for i in range(0,number_of_pts-1):
		frame = cv2.line(frame, imgpts[i], imgpts[4], line_color, 15)

	cuboidpyramid = 'tirumalcuboidpyramid.jpg'
	cv2.imwrite(cuboidpyramid,frame)
	print("Computed CUBOID PYRAMID")

def cuboid(worldpts):
	frame = cv2.imread(worldimg)
	imgpts = []
	xscale=1000
	yscale=1000
	for pt in worldpts:
		wpt = [pt[0],pt[1],pt[2],1]
		wpt = np.array(wpt)
		ipt = np.matmul(P,wpt)
		ipt = ipt/ipt[2]
		iptx = int(ipt[0]*xscale)
		ipty = int(ipt[1]*yscale)
		imgpts.append([iptx,ipty])
		frame = cv2.circle(frame, (iptx, ipty), 15, (0, 0, 255), 15)

	number_of_pts = len(imgpts)

	line_color = (0,0,0)
	# line_color = (0,255,255)


	## Filling with color
	pts1 = np.array([imgpts[0],imgpts[1],imgpts[2],imgpts[3]])
	frame = cv2.fillPoly(frame,pts=[pts1],color = (255,153,51))

	## Filling with color
	pts1 = np.array([imgpts[4],imgpts[5],imgpts[6],imgpts[7]])
	frame = cv2.fillPoly(frame,pts=[pts1],color = (102,255,178))


	pts1 = np.array([imgpts[0],imgpts[1],imgpts[4]])
	# frame = cv2.fillPoly(frame,pts=[pts1],color = (102,255,178))

	pts1 = np.array([imgpts[1],imgpts[2],imgpts[4]])
	# frame = cv2.fillPoly(frame,pts=[pts1],color = (102,255,178))

	pts1 = np.array([imgpts[2],imgpts[3],imgpts[4]])
	# frame = cv2.fillPoly(frame,pts=[pts1],color = (102,255,178))

	pts1 = np.array([imgpts[3],imgpts[0],imgpts[4]])
	# frame = cv2.fillPoly(frame,pts=[pts1],color = (102,255,178))

	## create base (draw lines)
	for i in range(0,4-1):
		frame = cv2.line(frame, imgpts[i], imgpts[i+1], line_color, 15)
	frame = cv2.line(frame, imgpts[0], imgpts[3], line_color, 15)

	## create upperlayer (draw lines)
	for i in range(4,8-1):
		frame = cv2.line(frame, imgpts[i], imgpts[i+1], line_color, 15)
	frame = cv2.line(frame, imgpts[4], imgpts[7], line_color, 15)

	## create sides (draw lines)
	for i in range(0,4):
		frame = cv2.line(frame, imgpts[i], imgpts[i+4], line_color, 15)

	cuboidop = 'tirumalcuboid.jpg'
	cv2.imwrite(cuboidop,frame)
	print("Computed CUBOID")

def cube(worldpts):
	frame = cv2.imread(worldimg)
	imgpts = []
	xscale=1000
	yscale=1000
	for pt in worldpts:
		wpt = [pt[0],pt[1],pt[2],1]
		wpt = np.array(wpt)
		ipt = np.matmul(P,wpt)
		ipt = ipt/ipt[2]
		iptx = int(ipt[0]*xscale)
		ipty = int(ipt[1]*yscale)
		imgpts.append([iptx,ipty])
		frame = cv2.circle(frame, (iptx, ipty), 15, (0, 0, 255), 15)

	number_of_pts = len(imgpts)

	line_color = (0,0,0)
	# line_color = (0,255,255)


	## Filling with color
	pts1 = np.array([imgpts[0],imgpts[1],imgpts[2],imgpts[3]])
	frame = cv2.fillPoly(frame,pts=[pts1],color = (255,153,51))

	## Filling with color
	pts1 = np.array([imgpts[4],imgpts[5],imgpts[6],imgpts[7]])
	frame = cv2.fillPoly(frame,pts=[pts1],color = (102,255,178))


	pts1 = np.array([imgpts[0],imgpts[1],imgpts[4]])
	# frame = cv2.fillPoly(frame,pts=[pts1],color = (102,255,178))

	pts1 = np.array([imgpts[1],imgpts[2],imgpts[4]])
	# frame = cv2.fillPoly(frame,pts=[pts1],color = (102,255,178))

	pts1 = np.array([imgpts[2],imgpts[3],imgpts[4]])
	# frame = cv2.fillPoly(frame,pts=[pts1],color = (102,255,178))

	pts1 = np.array([imgpts[3],imgpts[0],imgpts[4]])
	# frame = cv2.fillPoly(frame,pts=[pts1],color = (102,255,178))

	## create base (draw lines)
	for i in range(0,4-1):
		frame = cv2.line(frame, imgpts[i], imgpts[i+1], line_color, 15)
	frame = cv2.line(frame, imgpts[0], imgpts[3], line_color, 15)

	## create upperlayer (draw lines)
	for i in range(4,8-1):
		frame = cv2.line(frame, imgpts[i], imgpts[i+1], line_color, 15)
	frame = cv2.line(frame, imgpts[4], imgpts[7], line_color, 15)

	## create sides (draw lines)
	for i in range(0,4):
		frame = cv2.line(frame, imgpts[i], imgpts[i+4], line_color, 15)

	cubeop = 'tirumalcube.jpg'
	cv2.imwrite(cubeop,frame)
	print("Computed CUBE")


def pyramid(worldpts):
	frame = cv2.imread(worldimg)
	imgpts = []
	xscale=1000
	yscale=1000
	for pt in worldpts:
		wpt = [pt[0],pt[1],pt[2],1]
		wpt = np.array(wpt)
		ipt = np.matmul(P,wpt)
		ipt = ipt/ipt[2]
		iptx = int(ipt[0]*xscale)
		ipty = int(ipt[1]*yscale)
		imgpts.append([iptx,ipty])
		frame = cv2.circle(frame, (iptx, ipty), 15, (0, 0, 255), 15)

	number_of_pts = len(imgpts)

	line_color = (0,0,0)
	# line_color = (0,255,255)


	## Filling with color
	pts1 = np.array([imgpts[0],imgpts[1],imgpts[2],imgpts[3]])
	frame = cv2.fillPoly(frame,pts=[pts1],color = (255,153,51))

	pts1 = np.array([imgpts[0],imgpts[1],imgpts[4]])
	# frame = cv2.fillPoly(frame,pts=[pts1],color = (102,255,178))

	pts1 = np.array([imgpts[1],imgpts[2],imgpts[4]])
	frame = cv2.fillPoly(frame,pts=[pts1],color = (102,255,178))

	pts1 = np.array([imgpts[2],imgpts[3],imgpts[4]])
	# frame = cv2.fillPoly(frame,pts=[pts1],color = (102,255,178))

	pts1 = np.array([imgpts[3],imgpts[0],imgpts[4]])
	frame = cv2.fillPoly(frame,pts=[pts1],color = (102,255,178))

	## create base (draw lines)
	for i in range(0,number_of_pts-1):
		frame = cv2.line(frame, imgpts[i], imgpts[i+1], line_color, 15)
	frame = cv2.line(frame, imgpts[0], imgpts[3], line_color, 15)

	## create upper part of pyramid (draw lines)
	for i in range(0,number_of_pts-1):
		frame = cv2.line(frame, imgpts[i], imgpts[4], line_color, 15)

	pyramidop = 'tirumalpyramid.jpg'
	cv2.imwrite(pyramidop,frame)
	print("Computed PYRAMID")


def get_line(x1,y1,x2,y2):
	icap = y1-y2
	jcap = x2-x1
	kcap = (x1*y2)-(x2*y1)
	return icap,jcap,kcap

def get_point(ai,aj,ak,bi,bj,bk):
	icap = (aj*bk)-(ak*bj)
	jcap = (ak*bi)-(ai*bk)
	kcap = (ai*bj)-(bi*aj)
	return icap,jcap,kcap

def get_matrix(final_points):
	mat = []
	for pt in final_points:
		pt1 = pt[0]
		pt2 = pt[1]
		x1 = pt1[0]
		y1 = pt1[1]
		z1 = pt1[2]
		x2 = pt2[0]
		y2 = pt2[1]
		z2 = pt2[2]
		t = []
		t.append(x1*x2)
		t.append((x1*z2)+(x2*z1))
		t.append(y1*y2)
		t.append((y2*z1)+(y1*z2))
		t.append(z1*z2)
		# print(t)
		mat.append(t)
	final_mat = np.array(mat)
	# print(final_mat)
	return final_mat


img_points = [[(1227,794,1),(3008,758,1),(1009,1948,1),(3339,1909,1)],
            [(1066,978,1),(2849,564,1),(963,2308,1),(2892,2543,1)],
            [(1209,579,1),(2790,1016,1),(1163,2355,1),(2878,2223,1)],
            [(559,1143,1),(3362,1019,1),(1073,2392,1),(3130,2405,1)],
            [(640,866,1),(2818,600,1),(1074,2182,1),(2984,2532,1)],
            [(1864,462,1),(3311,1254,1),(1684,1993,1),(3525,2502,1)]]


final_points = []

for imgpt in img_points:
	pt1 = imgpt[0]
	pt2 = imgpt[1]
	pt3 = imgpt[2]
	pt4 = imgpt[3]

	# fi,fj,fk = get_line(pt1[0],pt1[1],pt2[0],pt2[1])
	# si,sj,sk = get_line(pt3[0],pt3[1],pt4[0],pt4[1])
	# v1i,v1j,v1k = get_point(fi,fj,fk,si,sj,sk)
	# v1i,v1j,v1k = get_line(fi/fk,fj/fk,si/sk,sj/sk)
	## ---- 

	fi,fj,fk = np.cross([pt1[0],pt1[1],1],[pt2[0],pt2[1],1])
	si,sj,sk = np.cross([pt3[0],pt3[1],1],[pt4[0],pt4[1],1])
	# v1i,v1j,v1k = np.cross([fi,fj,fk],[si,sj,sk])
	v1i,v1j,v1k = np.cross([fi/fj,1,fk/fj],[si/sj,1,sk/sj])

	fi,fj,fk = np.cross([pt1[0],pt1[1],1],[pt3[0],pt3[1],1])
	si,sj,sk = np.cross([pt2[0],pt2[1],1],[pt4[0],pt4[1],1])
	v2i,v2j,v2k = np.cross([fi/fj,1,fk/fj],[si/sj,1,sk/sj])
	# v2i,v2j,v2k = np.cross([fi,fj,fk],[si,sj,sk])
	# v2i,v2j,v2k = np.cross(fi/fk,fj/fk,si/sk,sj/sk)

	temp = []
	temp.append([v1i,v1j,v1k])
	temp.append([v2i,v2j,v2k])
	# print("vansishing points are: ",temp)
	final_points.append(temp)


A = get_matrix(final_points)

# print(A)

u,d,v=np.linalg.svd(A)
# print(v[4][:])
# print(v[-1,:])
w11,w13,w22,w23,w33=v[-1,:]

# W=[[w11,0,w13],[0, w22, w23],[w13, w23, w33]]
W=[[w11,0,w13],[0, w22, w23],[w13, w23, w33]]/w33
# print(W)

B = np.linalg.cholesky(W)
Bt = np.transpose(B)
K = np.linalg.inv(Bt)
K = K/K[2][2]
print("*****  K *******")
print(K)
print("************")

Xtrans = [
[0,0,9,1],
[7,0,9,1],
[7,5,9,1],
[0,5,9,1],
[3,3,9,1]
]

xtrans = [
[926/1000,2410/1000,1],
[3318/1000,2355/1000,1],
[3309/1000,643/1000,1],
[881/1000,673/1000,1],
[1943/1000,1359/1000,1]
]


X = np.transpose(np.array(Xtrans))
x = np.transpose(np.array(xtrans))

noofpts = len(Xtrans)

Kinv = np.linalg.inv(K)
Kinvx = np.dot(Kinv,x)

# print("*****  Kinvx *******")
# print(Kinvx)
# print("************")

size = (3*noofpts,12)
Rt_mat = np.zeros(size).astype(np.float32)
# print(Rt_mat)

i=0

while i<(noofpts):
	pt = Xtrans[i]
	Rt_mat[3*i][0]=pt[0]
	Rt_mat[3*i+1][3]=pt[0]
	Rt_mat[3*i+2][6]=pt[0]

	Rt_mat[3*i][1]=pt[1]
	Rt_mat[3*i+1][4]=pt[1]
	Rt_mat[3*i+2][7]=pt[1]

	Rt_mat[3*i][2]=pt[2]
	Rt_mat[3*i+1][5]=pt[2]
	Rt_mat[3*i+2][8]=pt[2]

	Rt_mat[3*i][9]=1
	Rt_mat[3*i+1][10]=1
	Rt_mat[3*i+2][11]=1

	i += 1

B = []
for c in range(len(Kinvx[0])):
	for r in range(len(Kinvx)):
		B.append([Kinvx[r][c]])

B = np.array(B)
# print(B)
# print(B.shape)
# print(Rt_mat)

Rtcol = np.linalg.lstsq(Rt_mat, B)[0]

# print("***** A *******")
# print(Rt_mat)
# print("************")

# print("***** B *******")
# print(B)
# print("************")

# print(Rtcol)

fin_Rt = [
[Rtcol[0][0],Rtcol[1][0],Rtcol[2][0],Rtcol[9][0]],
[Rtcol[3][0],Rtcol[4][0],Rtcol[5][0],Rtcol[10][0]],
[Rtcol[6][0],Rtcol[7][0],Rtcol[8][0],Rtcol[11][0]]
]

fin_Rt = (np.array(fin_Rt)).astype(np.float32)
# print(fin_Rt)
# print(fin_Rt.shape)



P = np.dot(K,fin_Rt)
P = P/P[2][3]
print("***** P *******")
print(P)
print("************")

worldimg = 'tirumal.jpg'

# pyramid([[1,2,9],[3,2,9],[3,4,9],[1,4,9],[2,3,5]])

# cube([[1,2,9],[3,2,9],[3,4,9],[1,4,9],[1,2,9-2],[3,2,9-2],[3,4,9-2],[1,4,9-2]])

# cuboid([[1,2,9],[5,2,9],[5,4,9],[1,4,9],[1,2,9-2],[5,2,9-2],[5,4,9-2],[1,4,9-2]])

cuboidpyramid([[0,0,9],[4,0,9],[4,2,9],[0,2,9],[1,1,9-1],[5,1,9-1],[5,3,9-1],[1,3,9-1],[3,2,9-5]])

