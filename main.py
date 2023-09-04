
from tkinter import *

import tkinter as tk
import tob_cnn as tobcn
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename

# Here, we are creating our class, Window, and inheriting from the Frame
# class. Frame is a class from the tkinter module. (see Lib/tkinter/__init__)
class Window(Frame):

    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None):
        
        # parameters that you want to send through the Frame class. 
        Frame.__init__(self, master)   

        #reference to the master widget, which is the tk window                 
        self.master = master

        #with that, we want to then run init_window, which doesn't yet exist
        self.init_window()

    #Creation of init_window
    def init_window(self):

        # changing the title of our master widget      
        self.master.title("GUI")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        w2 = Label(self, text="Tobacco Plant Detection",bg="orange"  ,fg="white"  ,width=30  ,height=3,font=('times', 30, 'italic bold underline'))
        w2.place(x=400,y=10)
        
        '''

        l=Button(self,text="Build Training Model", command=self.buildModel, bg="red"  ,fg="white"  ,width=20  ,height=1,font=('times', 20, 'italic bold underline'))
        l.place(x=200,y=200)
        
        '''
        
        k=Button(self,text="Browse For Input Image", command=self.showImgg, bg="red"  ,fg="white"  ,width=20  ,height=1,font=('times', 20, 'italic bold underline'))
        k.place(x=700,y=200)
        
        t=Button(self,text="Segment", command=self.SegmentImgg, bg="red"  ,fg="white"  ,width=20  ,height=1,font=('times', 20, 'italic bold underline'))
        t.place(x=350,y=600)
        
        t=Button(self,text="Classify Test Image", command=self.Classify, bg="red"  ,fg="white"  ,width=20  ,height=1,font=('times', 20, 'italic bold underline'))
        t.place(x=800,y=600)
        
        
    def buildModel(self):
        print('Images to be loaded here to build model')
        tobcn.traincnn()
        
    def Classify(self):
        print("Classify the Test Image")
        from keras.models import load_model
        model = load_model('tob_model.h5')
        
        #Compiling the model
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        (model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']))
        #Making New Prediction
        import numpy as np
        from keras.preprocessing import image
        
        test_image = image.load_img(self.load,target_size = (64,64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image,axis = 0)
        result = model.predict(test_image)
        if result[0][0]==1:
            a="Tobacco"
            print("Classified Output is Tobacco")
        else :
            a="NonTobacco"
            print("Classified Output is NonTobacco")

        s=Label(self,text=a,font=("arial",35),height = 1, width = 10)
        s.place(x=800,y=400)
        

    def showImgg(self):
        self.load = askopenfilename(filetypes=[("Image File",'.jpg')])
        
        im = Image.open(self.load)
    
        render = ImageTk.PhotoImage(im)

        # labels can be text or images
        img = Label(self, image=render,width=400,height=200)
        img.image = render
        img.place(x=200, y=350)
        
    def image_show(image, nrows=1, ncols=1, cmap='gray'):
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        return fig, ax
    
    
    def SegmentImgg(self):
        from skimage.feature import peak_local_max
        from skimage.morphology import watershed
        from scipy import ndimage
        import numpy as np
        import argparse
        import imutils
        import cv2
        
               
        image = cv2.imread(self.load)
        
        shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
        cv2.imshow("Input", image)
        cv2.imwrite('InputImage.jpg', image)
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cv2.imshow("Thresh", thresh)
        cv2.imwrite('ThreshImage.jpg', thresh)
           
        
        # find contours in the thresholded image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        print("[INFO] {} unique contours found".format(len(cnts)))
        # loop over the contours
        for (i, c) in enumerate(cnts):
            # draw the contour
            ((x, y), _) = cv2.minEnclosingCircle(c)
            cv2.putText(image, "#{}".format(i + 1), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.imshow("Watershed", image)
        cv2.imwrite('Watershed.jpg', image)
                
        D = ndimage.distance_transform_edt(thresh)
        localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh)
        # perform a connected component analysis on the local peaks,
        # using 8-connectivity, then appy the Watershed algorithm
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=thresh)
        print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
        
        for label in np.unique(labels):
            if label == 0:
                continue
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == label] = 255
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
            cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # show the output image
        cv2.imshow("Segment", image)
        cv2.imwrite('Segment.jpg', image)
        cv2.waitKey(0)
        
        '''
        render = ImageTk.PhotoImage(image)

        # labels can be text or images
        img = Label(self, image=render,width=400,height=200)
        img.image = render
        img.place(x=200, y=350)
        '''
        
        
                
# root window created. Here, that would be the only window, but
# you can later have windows within windows.
root = tk.Tk()

root.geometry("1300x800")
root.configure(background='blue')

#creation of an instance
app = Window(root)


#mainloop 
root.mainloop() 