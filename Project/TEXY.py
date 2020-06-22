import cv2
import time 
import tkinter as tk
import numpy as np
from PIL import Image,ImageTk
# 開啟影片檔案
def video_loop():
    cap = cv2.VideoCapture('C:/Users/loe_lin/Desktop/video/GOPR2319.MP4')

    def canny(img):
        img = cv2.Canny(img , 150 , 230)
        return img
    # def line_detect_possible_demo(image):
    #   lines = cv2.HoughLinesP(Home_plate_edge, 1, np.pi / 180, 60, minLineLength=60, maxLineGap=5)
    #   for(x1, y1), (x2, y2) in lines:
    #     if(y1 > 203 and y1 < 400):
    #         cv2.line(Home_plate_edge, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #   return Home_plate_edge


    # 以迴圈從影片檔案讀取影格，並顯示出來
    while(cap.isOpened()):
      ret, frame = cap.read()
      img = frame.copy()
      time.sleep(0.05) 
      # h, w = frame.shape[:2]
      gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
    
      ret1, otsu = cv2.threshold(gray,255,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
      kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
      opening  = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel, iterations=2)
      dst = cv2.cornerHarris(opening,2,3,0.04)
      dst = cv2.dilate(dst,None)
      print(dst)
      img[dst>0.01*dst.max()]=[0,0,255]
      # ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
      # dst = np.uint8(dst)

      # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

      # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
      # corners = cv2.cornerSubPix(gray,np.float32(centroids),(10,10),(-1,-1),criteria)

      # res = np.hstack((centroids,corners))

      # res = np.int0(res)
      # img[res[:,1],res[:,0]]=[0,0,255]
      # img[res[:,3],res[:,2]] = [0,255,0]
      # line_detect_possible_demo(frame)


      # contours, hierarchy = cv2.findContours(otsu,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
      # cnt = contours[0]
      # hull = cv2.convexHull(cnt)
      # cv2.drawContours(frame,[hull],0,(255,0,0),-1)

      Home_plate_edge = canny(otsu)
      # lines = cv2.HoughLinesP(Home_plate_edge, 1, np.pi / 180, 60, minLineLength=60, maxLineGap=5)
      # print(lines)
      # for line in lines:
      #   x1, y1, x2, y2 = line[0]
      #   if(y1 > 203 and y1 < 400):
      #       cv2.line(Home_plate_edge, (x1, y1), (x2, y2), (0, 0, 255), 2)
      # return Home_plate_edge    
      

      cv2.imshow('frame',frame)
      cv2.imshow('opening',opening)
      cv2.imshow('img',img)
      cv2.imshow('Home_plate_edge',Home_plate_edge)
    
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap.release()
    cv2.destroyAllWindows()


window = tk.Tk()
window.title('好球帶辨識系統')
window.geometry('1000x800')
window.configure(background='white')

header_label = tk.Label(window, text='請輸入個人資料')
header_label.pack()

strike_zone_up = tk.Frame(window)
strike_zone_up.pack(side=tk.TOP)
strike_zone_up_label = tk.Label(strike_zone_up, text='中間平行線高度')
strike_zone_up_label.pack(side=tk.LEFT)
strike_zone_up_entry = tk.Entry(strike_zone_up)
strike_zone_up_entry.pack(side=tk.LEFT)

strike_zone_down = tk.Frame(window)
strike_zone_down.pack(side=tk.TOP)
strike_zone_down_label = tk.Label(strike_zone_down, text='膝蓋下緣高度')
strike_zone_down_label.pack(side=tk.LEFT)
strike_zone_down_entry = tk.Entry(strike_zone_down)
strike_zone_down_entry.pack(side=tk.LEFT)

calculate_btn = tk.Button(window, text='馬上計算', command=video_loop)
calculate_btn.pack()

window.mainloop()