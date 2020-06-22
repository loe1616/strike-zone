import cv2
import time 
import tkinter as tk
import numpy as np    
    
cap = cv2.VideoCapture('C:/Users/loe_lin/Desktop/video/GOPR2318.mp4')

    # 以迴圈從影片檔案讀取影格，並顯示出來
while(cap.isOpened()):
  ret, frame = cap.read()
  img = frame.copy()
  time.sleep(0.05) 

  gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 

  ret1, otsu = cv2.threshold(gray,255,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    

  HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
  lower_green = np.array([0, 0, 221])
  upper_green = np.array([180, 30, 255])
  mask = cv2.inRange(HSV, lower_green, upper_green)
  res = cv2.bitwise_and(frame, frame, mask=mask)

  h, w = frame.shape[:2]
  y, x = np.mgrid[1:h:1, 1:w:1].reshape(2,-1).astype(int)
  lines = np.vstack([x, y]).T.reshape(-1, 2)
    # cv2.polylines(frame, lines, 0, (0, 255, 0))
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
  dilate = cv2.dilate(otsu,kernel,iterations = 1)
  opening  = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel, iterations=2)
    # print(opening)

  Home_plate_edge = cv2.Canny(otsu , 13 , 200)
  # cv2.imshow('opening', opening )
#   for (x1, y1) in lines:
#     if Home_plate_edge[y1][x1] > 1 and opening[y1][x1] > 1 and y1 > 550:
#       mylist1.append(x1)
#       mylist2.append(y1)
      # cv2.circle(img, (x1, y1), 1, (0, 255, 0), -1)

      

  cv2.imshow('frame',frame)
  cv2.imshow('HSV',HSV)
  cv2.imshow('res',res)
  cv2.imshow('otsu',otsu)
  cv2.imshow('img',img)
  cv2.imshow('Home_plate_edge',Home_plate_edge)
  cv2.imshow('dilate',dilate)
    
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()