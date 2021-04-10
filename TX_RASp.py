import cv2;
import numpy as np;
import math;
from scipy import ndimage;
from socket import *;

host, port = '0.0.0.0', 12345;

sok = socket(AF_INET, SOCK_STREAM);
sok.bind((host,port));
sok.listen(5);

face_cascade = cv2.CascadeClassifier('E:\data\opencv\sources\data\haarcascades\haarcascade_frontalface_alt.xml');

basePath = "E:\\projpython\\";
svm = cv2.SVM()
svm.load(basePath+'T_Data.dat');

def retlbl(label):
    if label == 1.0:
        return 'Forward';
    elif label == 2.0:
        return 'Stop';
    elif label == 3.0:
        return 'Reverse';
    #elif label == 3.0:
    #   return 'Zero';


def findFeatures(img,cnt, count):
    hull = cv2.convexHull(cnt);
    c_area = cv2.contourArea(cnt);
    h_area = cv2.contourArea(hull);
    x,y,w,h = cv2.boundingRect(cnt);
    M = cv2.moments(img);
    HuM = cv2.HuMoments(M);	
    if cnt == None:
        return np.zeros((1,4),dtype = np.float32).flatten();
    AUX_RATIO = float(w)/h;
    R_C_AREA = float(w*h)/c_area;
    R_H_AREA = float(w*h)/h_area;
    SOLIDITY = float(c_area)/h_area;
    FV = np.asarray(np.r_[count, AUX_RATIO, R_C_AREA, R_H_AREA, SOLIDITY, HuM.flatten()],dtype = np.float32);
    return FV;

def findConvDefects(im,cnt,hull,distance):
    imtl = np.copy(im);
    defects = cv2.convexityDefects(cnt,hull);
    count = 0;
    dn = [];
    for d in defects:
        s,e,f,dist = d.flatten();
        start,end,far = tuple(cnt[s][0]),tuple(cnt[e][0]),tuple(cnt[f][0]);
        cv2.line(imtl,start,end,(0,0,255),2);
        if dist >= distance:
            cv2.circle(imtl,far,3,(0,255,255),-1);
            cv2.line(imtl,tuple((np.add(start,end)/2)),far,(0,255,0),1);
            count = count+1;
            dn.append(dist);
    return imtl,count,dn;


def faceSubtraction(im):
        imr = np.copy(im);
        img = cv2.cvtColor(imr,cv2.COLOR_BGR2GRAY);
        face = face_cascade.detectMultiScale(im,1.3,5);
        for (x,y,w,h) in face:
                imr[y:y+h,x:x+w] = 0;
        return imr;

def skinColor(im):
        imc = np.zeros(im[...,0].shape,dtype=np.uint8);
        B,G,R = im[...,0], im[...,1], im[...,2];
        cond = (R > 95)*(G>40)*(B>20)*((np.max(im,axis=2)-np.min(im,axis=2))>15)*(abs(R-G)>15)*(R>G)*(R>B);
        imc[cond] = 255;
        return imc;

def skinColor1(im):
        imc = np.zeros(im[...,0].shape,dtype=np.uint8);
        hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV);
        H,S,V = hsv[...,0], hsv[...,1], hsv[...,2];
        cond = (H<20)*(50<S)*(S<150)*(68<V);
        imc[cond] = 255;
        return imc      

def skinColor2(im):
        imc =  np.zeros(im[...,0].shape,dtype=np.uint8);
        YCrCb = cv2.cvtColor(im,cv2.COLOR_BGR2YCR_CB);
        Y,Cr,Cb = YCrCb[...,0], YCrCb[...,1], YCrCb[...,2];
        cond = (135<Cr)*(Cr<183)*(120<Cb)*(Cb<154);
        imc[cond] = 255;
        return imc;

def skinColorCom(hsv,ycbcr):
        im = np.zeros(hsv.shape,dtype=np.uint8);
        im[(hsv == 255)*(ycbcr == 255)] = 255;
        return im;

def morphOperation( imr):
        dil = np.ones((5,5), dtype = np.uint8);
        erd = np.ones((5,5), dtype = np.uint8);
        imd = cv2.dilate(imr, dil, iterations = 1);
        imed = cv2.erode(imd, erd, iterations = 1);
        return  imed;         

def handConnectLabel(im, imr):
        se = np.ones((9,9), dtype = np.uint8);
        imcl, numl = ndimage.label(imr, structure = None);
        imtemp = np.zeros(imcl.shape, dtype = np.uint8);
        imh = np.zeros(im.shape, dtype = np.uint8);
        for i in xrange(numl):
                if i == 0:
                        continue;
                imtemp = np.zeros(imcl.shape, dtype = np.uint8);
                imtemp[imcl == i] = 255;
                if np.sum(imtemp) > 255*10000:
                        imh[imtemp == 255] = im[imtemp == 255];
                        break;
        return imh, imtemp;

def findContour(im,thr,Max_Area):
        imth = np.copy(im);
        thr1 = np.copy(thr);
        cnt = None;
        contour,hierarchy = cv2.findContours(thr1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE);
        for i in contour:
                area = cv2.contourArea(i);
                if area >= 10000:
                        cnt = i
        if cnt == None:
                return None,None,None;
        
        hull1 = cv2.convexHull(cnt,returnPoints = False);
        hull2 = cv2.convexHull(cnt);


        l,count,d = findConvDefects(imth,cnt,hull1,2500)
        fv = findFeatures(thr,cnt,count);
        label = svm.predict(fv);
        st = retlbl(label);
        cv2.drawContours(imth,contour,-1,(0,0,255),2);
            
        
        return st,imth,cnt;


def isHand(cnt):
        th = 2;
        if th < 10:
                return True;
        return False;
def isHand1(cnt):
    if cnt == None:
        #print 'None';
        return False;
    else:
        spcon1 = np.load('cnt.npy');
        spcon2 = np.load('cnt1.npy');
        #print 20*'==';print spcon1,spcon2;print 20*'##';
        ret = min(cv2.matchShapes(cnt,spcon1,1,0.0),cv2.matchShapes(cnt,spcon2,1,0.0));
        print ret;
        area = cv2.contourArea(cnt);
        if (ret <= 0.11)*(area >= 10000) :
            return True;
        else:
            return False;

        
def boundingBox(imr,found):
        im = faceSubtraction(imr);
        img = morphOperation(skinColor2(im));
        #cv2.imshow('Video',im);
        #cv2.imshow('RGB',img);
        Roi3,Roibw = handConnectLabel(imr,img);
        #cv2.imshow('Color',Roi3);
        #cv2.imshow('BW',Roibw);
        if not found:
                st,imc,contour = findContour(imr,Roibw,0);
                if imc != None:
                        #cv2.imshow('All Cotours',imc);
                        if isHand(contour):
                                if cv2.waitKey(1) == ord('u'):
                                    np.save('cnt1.npy',contour);
                                    print 'Contour stored';
                                
                                #isHand1(contour);
                                x,y,w,h = cv2.boundingRect(contour);
                                direc = 0;
                                if st == 'Forward':
                                    direc = Orientation(imr,contour);
                                cv2.imshow('contour',imc);
                                return st,direc,(x,y,w,h),Roi3;
        return -1,-1,-1,Roi3;

def initCamShift(im,(x,y,w,h)):
        hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV);
        mask = np.zeros(im[...,0].shape,dtype=np.uint8);
        mask[y:y+h,x:x+w] = 1;
        roi_hist = cv2.calcHist([hsv],[0],mask,[180],[0,180]);
        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
        return roi_hist;


def Orientation(im,cnt):
    imt = np.copy(im);
    rows, cols = imt.shape[:2];
    (vx,vy,x,y) = cv2.fitLine(cnt,cv2.NORM_L2SQR,0,0.01,0.01);
    ly = (-x*vy/vx + y);
    ry = ((cols-x)*vy/vx + y);
    ang = math.atan((ry-ly)/(cols-1))*180/math.pi;
    cv2.putText(imt,str(ang),(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2);
    cv2.line(imt,(0,ly),(cols-1,ry),255,2);
    cv2.imshow('Angle',imt);
    if (-90 <= ang)*(ang <= -60)+(60 <= ang)*(ang <= 90):
        direc = 0;
    elif ang < 0:
        direc = -1;
    else:
        direc = 1;
    return direc;
        
cam = cv2.VideoCapture(0);
found = False;
#fcc = cv2.VideoWriter_fourcc(*'XVID');
#vid = cv2.VideoWriter('HG1.avi',-1,20,(640,480));

k = 1;
while True:
        rt,imr = cam.read();
        imsv = np.copy(imr);
        #print q
        #q.send('Hello');
        if rt:
                t1 = cv2.getTickCount();
                st,direc,data,Roi = boundingBox(imr,found);
                if st == -1:
                    st = '';
                st1 = '';
                if data != -1:
                        x,y,w,h = data;
                        if st == 'Forward':
                            if direc == -1:
                                st1 = 'Left';
                                q,addr = sok.accept();
                                q.send('Forward Left');
                            elif direc == 1:
                                st1 = 'Right';
                                q,addr = sok.accept();
                                q.send('Forward Right');
                            else:
                                st1 = 'Straight';
                                q,addr = sok.accept();
                                q.send('Forward Straight');
                        elif st == 'Reverse':
                            q,addr = sok.accept();
                            q.send('Reverse')
                        else:
                            q,addr = sok.accept();
                            q.send('Stop');
                        cv2.rectangle(imr,(x,y),(x+w,y+h),(0,0,255),2);
                        cv2.putText(imr,st+' '+st1,(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2);
                else:
                        q,addr = sok.accept();
                        q.send('Stop');
                        
                cv2.imshow('Hand',imr);
                #cv2.imwrite(basePath+'rasp\\'+str(k)+'.jpg',imsv);
                k = k+1;
                t2 = cv2.getTickCount();
                #print (t2-t1)/cv2.getTickFrequency();
        
        #q.send(st+st1);
        if cv2.waitKey(1) == ord('q'):
                break;

cv2.destroyAllWindows();
cam.release();
