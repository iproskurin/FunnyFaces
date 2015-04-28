# FINAL
# Ihor Proskurin

import numpy as np
import scipy as sp
import scipy.signal
import cv2
from sklearn.cluster import MiniBatchKMeans
import blend

# Import ORB as SIFT to avoid confusion.
try:
    from cv2 import ORB as SIFT
except ImportError:
    try:
        from cv2 import SIFT
    except ImportError:
        try:
            SIFT = cv2.ORB_create
        except:
            raise AttributeError("Your OpenCV(%s) doesn't have SIFT / ORB."
                                 % cv2.__version__)

def getEdges(image):
    output = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
    return 255-cv2.Canny(output, 200, 300, 3)

def smooth(image):
    output = image
    for i in range(20):
        output = cv2.bilateralFilter(output, 9, 9, 7)
    return output

def cartoonify(image, filename='test'):
    image = faceManip(image)
    #cv2.imwrite(filename+"image-faceManip.jpg", image)
    image = contrast(image, 13.5)
    #cv2.imwrite(filename+"image-contrast.jpg", image)
    image = smooth(image)
    #cv2.imwrite(filename+"image-smooth1.jpg", image)
    image = colorQuantization(image, 20)
    #cv2.imwrite(filename+"image-colorQuant.jpg", image)
    image = drawEdges(image)
    #cv2.imwrite(filename+"image-edges.jpg", image)
    image = smooth(image)
    #cv2.imwrite(filename+"image-smooth2.jpg", image)
    return image

def drawMask(image):
    mask = getEdges(image)
    return cv2.bitwise_and(image, image, mask = mask)

def drawEdges(image):
    mask = getEdges(image)
    cnt, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, cnt, -1, (90,90,90), 1, 8, hier, 5)
    return image

def colorQuantization(image, n_clusters):
    (h, w) = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    image = image.reshape((h, w, 3))

    # convert from L*a*b* to RGB
    return cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)

def saturate(image):
    return cv2.add(image, 50)

def contrast(image, alpha):
    output = alpha * image
    output = cv2.normalize(output.astype(int), None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(output)

def scaleElement(orig, x, y, w, h, scaleX, scaleY):
    object = orig[y:y+h, x:x+w]
    newW = int(w*scaleX)
    newH = int(h*scaleY)
    newX = max(int(x + w/2 - newW/2),0)
    newY = max(0, int(y + h/2 - newH/2))
    
    if (w==0 or y==0 or scaleX==0 or scaleY==0 or newH==0 or newW==0):
        return orig

    rect = np.array([
        [0, 0],
        [w-1, 0],
        [w-1, h-1],
        [0, h-1]], dtype = "float32")
        
    dst = np.array([
         [0, 0],
         [newW-1, 0],
         [newW-1, newH-1],
         [0, newH-1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    newObject = cv2.warpPerspective(object, M, (newW, newH))

    orig = blendObject(orig, newObject, newX, newY, newH, newW)

    return orig

def blendObject(orig, object, x, y, h, w):
    image1 = orig
    image2 = np.copy(orig)
    image2[y:y+h, x:x+w] = object
    mask = np.zeros(orig.shape, dtype=np.float)
    #cv2.imwrite(str(x)+"beforeBlend.jpg", image2)
    
    cv2.ellipse(mask, (x+w/2,y+h/2), (w/2, h/2), 0, 0, 360, (255,255,255), thickness=-1)
    orig = blend.run_blend(image1, image2, mask)
    return orig

def faceDetection(image, drawFeatures=False):
    face_cascade = cv2.CascadeClassifier('frontalface.xml')
    rightear_cascade = cv2.CascadeClassifier('rightear.xml')
    leftear_cascade = cv2.CascadeClassifier('leftear.xml')
    profileface_cascade = cv2.CascadeClassifier('profileface.xml')
    eye_cascade = cv2.CascadeClassifier('eye.xml')
    mouth_cascade = cv2.CascadeClassifier('mouth.xml')
    nose_cascade = cv2.CascadeClassifier('nose.xml')
    smile_cascade = cv2.CascadeClassifier('smile.xml')
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = eye_cascade.detectMultiScale(gray, 1.2, 10)
    noses = nose_cascade.detectMultiScale(gray, 1.3, 5)
    smiles = smile_cascade.detectMultiScale(gray, 1.9, 5)
    smiles = np.concatenate((smiles, mouth_cascade.detectMultiScale(gray)))
    
    faces, eyes, noses, smiles = faces[:1], eyes[:2], noses[:1], smiles[:1]
    
    if (drawFeatures):
        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        
        for (x,y,w,h) in eyes:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        
        for (x,y,w,h) in noses:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)

        for (x,y,w,h) in smiles:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,128,0),2)

    return (faces[:1], eyes[:2], noses[:1], smiles[:1], image)


def faceManip(image):
    (faces, eyes, noses, smiles, im) = faceDetection(image)

    stringOutput = ''
    
    for (x,y,w,h) in eyes:
        stringOutput += 'eyes bigger, '
        image = scaleElement(image, x, y, w, h, 1.35, 1.8)

    for (x,y,w,h) in noses:
        stringOutput += 'nose like potato, '
        image = scaleElement(image, x, y, w, h, 1.4, 1.5)

    for (x,y,w,h) in smiles:
        stringOutput += 'smile wider'
        image = scaleElement(image, x, y, w, h, 1.3, 1.2)

    for (x,y,w,h) in faces:
        if h > .98*w:
            stringOutput += 'face longer, '
            image = scaleElement(image, x, y, w, h, .95, 1.1)
        else:
            stringOutput += 'face wider, '
            image = scaleElement(image, x, y, w, h, 1.1, .95)

    print 'Changes applied: [', stringOutput, ']\n'
    return image