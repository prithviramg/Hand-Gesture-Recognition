import cv2;
import numpy as np;
from scipy import ndimage;

face_cascade = cv2.CascadeClassifier('E:\data\opencv\sources\data\haarcascades\haarcascade_frontalface_alt.xml');

def learnColor(im):
        img = cv2.cvtColor(im,cv2.COLOR_BGR2HSV);
        m = (np.mean(img[:,:,1]), np.mean(img[:,:,2]));
        v = (np.var(img[:,:,1]), np.var(img[:,:,2]));
        return m,v

def findHand(im,m,v):
        img = cv2.cvtColor(im,cv2.COLOR_BGR2HSV);
        imbw = mcr = mcb = vcr = vcb = np.zeros(img[:,:,0].shape, dtype = np.uint8);
        m1,m2 = abs(img[:,:,1]-m[0]), abs(img[:,:,2]-m[1])
        mcr[m1<1] =  mcb[m2<1] = 255;
        vcr[abs(m1*m1-v[0]) < 0.25] = vcb [abs(m2*m2-v[1]) < 0.25] = 255;
        imbw[(mcr==255)*(mcb==255)*(vcr==255)*(vcb==255)] = 255 ;
        imr = np.zeros(im.shape, dtype = np.uint8);
        imr[imbw == 255] = im[imbw == 255];
        return imr, imbw


def learnColor1(im):
        #img = cv2.cvtColor(im,cv2.COLOR_BGR2HSV);
        img = im;
        m = (np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2]));
        v = (np.var(img[:,:,0]), np.var(img[:,:,1]), np.var(img[:,:,2]));
        return m,v

def findHand1(im,m,v):
        #img = cv2.cvtColor(im,cv2.COLOR_BGR2HSV);
        img = im;
        imbw = my = mcr = mcb = vy = vcr = vcb = np.zeros(img[:,:,0].shape, dtype = np.uint8);
        m0,m1,m2 = abs(img[:,:,0]-m[0]), abs(img[:,:,1]-m[1]), abs(img[:,:,2]-m[2])
        my[m0<1] = mcr[m1<1] =  mcb[m2<1] = 255;
        vcr[abs(m0*m0-v[0]) < 0.25] = vcr[abs(m1*m1-v[1]) < 0.25] = vcb [abs(m2*m2-v[2]) < 0.25] = 255;
        imbw[(my==255)*(mcr==255)*(mcb==255)*(vy==255)*(vcr==255)*(vcb==255)] = 255 ;
        imr = np.zeros(im.shape, dtype = np.uint8);
        imr[imbw == 255] = im[imbw == 255];
        return imr, imbw




def morphOperation(im, imr):
        dil = np.ones((20,20), dtype = np.uint8);
        erd = np.ones((10,10), dtype = np.uint8);
        imd = cv2.dilate(imr, dil, iterations = 1);
        imed = cv2.erode(imd, erd, iterations = 1);
        imh = np.zeros(im.shape, dtype = np.uint8);
        imh[imed == 255] = im[imed == 255];
        return imh, imd, imed;

def faceSubtraction(im):
        imr = np.copy(im);
        img = cv2.cvtColor(imr,cv2.COLOR_BGR2GRAY);
        face = face_cascade.detectMultiScale(im,1.3,5);
        for (x,y,w,h) in face:
                imr[y:y+h,x:x+w] = 0;
        return imr;             

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
                if np.sum(imtemp) > 2000:
                        imh[imtemp == 255] = im[imtemp == 255];
                        break;
                else:
                        imtemp = None;
                        imh = None;
        return imh, imtemp;

        
cam = cv2.VideoCapture(0);

colorLearned = False;

while True:
        rt,im = cam.read();
        if rt:
                if colorLearned:
                        imf = faceSubtraction(im);
                        Hc, Hbw = findHand(imf,m,v);
                        Mo, d, e = morphOperation(im, Hbw);
                        roi, roibw = handConnectLabel(im, e);
                        if roi != None:
                                cv2.imshow('Hand',roi);
                if cv2.waitKey(1) == ord('l'):
                        m,v = learnColor(im[50:100,50:100]); print m,v;
                        colorLearned = True;
                        print 'Learning Over';
                cv2.rectangle(im,(50,50),(100,100),(0,0,255),2);
                cv2.imshow('Video',im);
                if cv2.waitKey(1) == ord('q'):
                        break;
cv2.destroyAllWindows();
cam.release();
                        
                                
