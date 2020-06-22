import cv2
import time 
import tkinter as tk
import numpy as np    


cap = cv2.VideoCapture('C:/Users/loe_lin/Desktop/video/GOPR2318.mp4')

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# videoWriter = cv2.VideoWriter('C:/Users/loe_lin/Desktop/video/123.mp4', fourcc, 120.0, (1280,720))
    # 以迴圈從影片檔案讀取影格，並顯示出來
ret, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

c = []



while(cap.isOpened()):
  
  ret, frame = cap.read()
  time.sleep(0.05) 

  gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 

  ret1, otsu = cv2.threshold(gray,255,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
  
#----------------------------------------------------------------------------------------------------------------------------
  h, w = frame.shape[:2]
  y, x = np.mgrid[1:h:1, 1:w:1].reshape(2,-1).astype(int)
  lines = np.vstack([x, y]).T.reshape(-1, 2)
    # cv2.polylines(frame, lines, 0, (0, 255, 0))
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
  opening  = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel, iterations=2)
    # print(opening)
  Home_plate_edge = cv2.Canny(otsu , 13 , 200)
#----------------------------------------------------------------------------------------------------------------------------  
  flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
  prevgray = gray

  fx1, fy1 = flow[:,:,0], flow[:,:,1]
  v = np.sqrt(fx1*fx1+fy1*fy1)
  ang = np.arctan2(fy1, fx1) + np.pi
    # print(ang.shape)
  hsv = np.zeros((h, w, 3), np.uint8)
  hsv[...,0] = ang*(180/np.pi/2)
  hsv[...,1] = 255
  hsv[...,2] = np.minimum(v*4, 255)
#----------------------------------------------------------------------------------------------------------------------------
  # cv2.imshow('opening', opening )
  mylist1 = []
  mylist2 = []
  mylist3 = []
  mylist4 = []
  for (x, y) in lines:  
    if Home_plate_edge[y][x] > 1 and opening[y][x] > 1 and y > 550:
      mylist1.append(x)
      mylist2.append(y) 
    if hsv[y][x][2] > 9:
      mylist3.append(x)
      mylist4.append(y)
      cv2.circle(frame, (x,y), 1, (255, 0, 0), -1)

    
  cv2.circle(frame, (max(mylist1), min(mylist2)), 1, (0, 255, 0), -1)
  cv2.circle(frame, (min(mylist1), min(mylist2)), 1, (0, 255, 0), -1)
  
  b1 = max(mylist1) - min(mylist1)
  G = (b1*140 - min(mylist1)) / b1
  Y = (b1*40 - min(mylist1)) / b1
    # print( b1,round(G))
    
  y1 = min(mylist2) - 2*int(Y)
  y2 = min(mylist2) - 2*int(G)
  x1 = min(mylist1)
  x2 = max(mylist1)

  
  cv2.line(frame, (x1, y1), (x1, y2), (0, 0, 255), 3)
  cv2.line(frame, (x2, y1), (x2, y2), (0, 0, 255), 3)
  cv2.line(frame, (x2, y2), (x1, y2), (0, 0, 255), 3)
  cv2.line(frame, (x2, y1), (x1,y1), (0, 0, 255), 3)
  # cv2.line(img, (300, 0), (300, 300), (0, 0, 255), 3)
  # cv2.line(img, (300,0), (600,0), (0, 0, 255), 3)
  # cv2.line(img, (300,300),(600,300), (0, 0, 255), 3)
  # cv2.line(img, (600,300),(600,0), (0, 0, 255), 3)
#----------------------------------------------------------------------------------------------------------------------------
  # print(mylist3,mylist4)
  # if len(mylist1) != 0:
  a = 0
  b = 0
  
  
  for i in mylist3:
    if x1 < i < x2:
      a = a + 1 
  for j in mylist4:
    if y2 < j < y1:
      b = b + 1  
  if a > 75 and b > 75:    
    c.append(1)
  else:
    c.append(0)
  if len(c) > 3:
    del(c[0])
  if c[0] == 1 and c[1] == 1 and c[2]==0:
    print('好球')
  else:
    print('壞球')
    # if y1 > 0:
    #   if hsv[y][x][2] > 1 :  
    #     cv2.circle(img, (x, y), 1, (255, 0, 0), -1)
  # print(min(mylist1),min(mylist2),max(mylist1), min(mylist2))         
#----------------------------------------------------------------------------------------------------------------------------
  # for (x, y) in lines:
  #   if hsv[y1][x1][2] > 1:
  #     cv2.circle(img, (x1, y1), 1, (255, 0, 0), -1)
      # if x2 > x < x1 and y1 > y < y2 :
      #   ball_range = ball_range + 1
      #   print(ball_range)
    
  # if hsv[2].all() > 1 :
  #   cv2.circle(img, (x1, y1), 1, (255, 0, 0), -1)
  # videoWriter.write(frame) 
  # print(img.shape)
  cv2.imshow('frame',frame)
  # print(frame.shape)
  # cv2.imshow('otsu',otsu)
  # cv2.imshow('img',img)
  # cv2.imshow('Home_plate_edge',Home_plate_edge)
    
  key = cv2.waitKey(1)
  if key == ord('q') or key == 27:
    break
  elif key == ord(' '):
    while(cv2.waitKey(1)!=ord(' ')):
      pass
cap.release()
# videoWriter.release()
cv2.destroyAllWindows()
