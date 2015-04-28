import sys
import os
import numpy as np
import cv2

import final

def test_cartoonify():
    cartoonify(image)

if __name__ == "__main__":
    sourcefolder = os.path.abspath(os.path.join(os.curdir, "images", "source"))
    outfolder = os.path.abspath(os.path.join(os.curdir, "images", "output"))

    print "Image source folder: {}".format(sourcefolder)
    print "Image output folder: {}".format(outfolder)

    print "Searching for folders with images in {}.".format(sourcefolder)

    # Extensions recognized by opencv
    exts = [".bmp", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".jpeg", ".jpg", 
            ".jpe", ".jp2", ".tiff", ".tif", ".png"]

    # For every image in the source directory
    for dirname, dirnames, filenames in os.walk(sourcefolder):
        for filename in filenames:
            name, ext = os.path.splitext(filename)
            if ext.lower() not in exts:
                continue
            filepath = os.path.join(dirname, filename)
            input = cv2.imread(filepath)

            if input == None:
                continue

            print "Cartoonifying... ", outfolder+filename
            setname = filename
            cartoonImage = final.cartoonify(input, filename)

            cv2.imwrite(os.path.join(outfolder, filename)+'.jpg', np.concatenate((input, cartoonImage), axis=1))
    print "DONE\n"
