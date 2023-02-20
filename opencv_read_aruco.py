import cv2
import cv2.aruco as aruco
import numpy as np
import os

def findArucoMarkers(img, markerSize=6, totalMarkers=250, draw=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParam)
    
    #print(ids)
    if draw:
        aruco.drawDetectedMarkers(img,bboxs)
        
    return [bboxs,ids]

def augmentAruco(bbox,id,img):
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]
    
    cx = int((tl[0] + br[0]) / 2.0)
    cy = int((tl[1] + br[1]) / 2.0)
    
    cv2.circle(img, (cx,cy),4, (0,0,255),-1)
    cv2.putText(img,str(id), (tl[0],tl[1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    
    #print(tl, tr, br, bl,id)
    
def main():
    cap = cv2.VideoCapture(0)
    
    while True:
        success, img = cap.read()
        arucoFound = findArucoMarkers(img)
        
        if len(arucoFound[0]) !=0:
            for bbox, id in zip(arucoFound[0],arucoFound[1]):
                x = augmentAruco(bbox,id,img)
                #print(bbox, id)
                
        cv2.imshow("Image",img)
        
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
    
if __name__ == "__main__":
    main()