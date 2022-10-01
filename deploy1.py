'''
CODER ZERO
connect with me at: https://www.youtube.com/channel/UCKipQAvBc7CWZaPib4y8Ajg
How to train custom yolov5: https://youtu.be/12UoOlsRwh8
DATASET: 1) https://www.kaggle.com/datasets/deepakat002/indian-vehicle-number-plate-yolo-annotation
         2) https://www.kaggle.com/datasets/elysian01/car-number-plate-detection
'''
### importing required libraries
from asyncio.windows_events import NULL
import tkinter
import torch
from tkinter import *
from datetime import date , datetime
import cv2
import time
import pyrebase
# import pytesseract
import re
import numpy as np
#import easyocr
import pandas as pd
from playsound import playsound
import json
import requests
from datetime import date
import time
import schedule
#import geocoder
#from openpyxl import load_workbook

##### DEFINING GLOBAL VARIABLE
# EASY_OCR = easyocr.Reader(['en']) ### initiating easyocr
# OCR_TH = 0.2
today = date.today()
now = datetime.now()

current_time = now.strftime("%H:%M:%S")
global out

config={  
  'apiKey': "AIzaSyCLscXyqA0KJCsUnebYEqn6TXLlA_HTc1g",
  'authDomain': "sampleproject-ef4d0.firebaseapp.com",
  'databaseURL': "https://sampleproject-ef4d0-default-rtdb.firebaseio.com",
  'projectId': "sampleproject-ef4d0",
  'storageBucket': "sampleproject-ef4d0.appspot.com",
  'messagingSenderId': "863988637736",
  'appId': "1:863988637736:web:25674b1d514d62e91b951d",
  'measurementId': "G-ZL0TYCYTXP"
}
firebase=pyrebase.initialize_app(config)
storage = firebase.storage()
database=firebase.database()
path_cloud = 'detected/'+str(today)+'.jpg'
path_local = str(today)+'.jpg'
### -------------------------------------- function to run detection ---------------------------------------------------------
def detectx (frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)
    # results.show()
    # print( results.xyxyn[0])
    # print(results.xyxyn[0][:, -1])
    # print(results.xyxyn[0][:, :-1])

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels , cordinates

### ------------------------------------ to plot the BBox and results --------------------------------------------------------
def plot_boxes(results, frame,classes):

    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels

    """
    labels, cord = results
    print(labels)
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f"[INFO] Total {n} detections. . . ")
    print(f"[INFO] Looping through all detections. . . ")


    ### looping through the detections
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.75: ### threshold value for detection. We are discarding everything below this value
            print(f"[INFO] Extracting BBox coordinates. . . ")
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
            text_d = classes[int(labels[i])]
            print(text_d)
            print(row[4])
            #ip = geocoder. ip("me")
            if(text_d in ['cow','dog','cat','sheep']):
                playsound('pu.mp3',False)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
                cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background
                cv2.putText(frame, f"{text_d + str(row[4])+ str(current_time) }", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 2)
                cv2.imwrite(str(today)+".jpg",frame)
                storage.child(path_cloud).put(path_local)
                # reader = pd.read_excel(r'demo.xlsx',engine='xlsxwriter')
                # if reader is NULL:
                #     df = pd.DataFrame({'Name': text_d,
                #     'location': ip})
                #     writer = pd.ExcelWriter('demo.xlsx', engine='xlsxwriter')
                #     writer.save()
                # writer = pd.ExcelWriter('demo.xlsx', engine='openpyxl')
                # writer.book = load_workbook('demo.xlsx')
                # writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
            
                # df.to_excel(writer,index=False,header=False,startrow=len(reader)+1)
                # writer.close()
                # writer = pd.ExcelWriter('demo.xlsx', engine='xlsxwriter')
                # df.to_excel(writer, sheet_name='Sheet1', index=False)
                # writer.save()

            # cv2.imwrite("./output/dp.jpg",frame[int(y1):int(y2), int(x1):int(x2)])

            coords = [x1,y1,x2,y2]

            #plate_num = recognize_plate_easyocr(img = frame, coords= coords, reader= EASY_OCR, region_threshold= OCR_TH)


            # if text_d == 'mask':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
            cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background
            cv2.putText(frame, f"{text_d + str(row[4]) }", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 2)

            # cv2.imwrite("./output/np.jpg",frame[int(y1)-25:int(y2)+25, int(x1)-25:int(x2)+25])




    return frame



#### ---------------------------- function to recognize license plate --------------------------------------


# function to recognize license plate numbers using Tesseract OCR
def recognize_plate_easyocr(img, coords,reader,region_threshold):
    # separate coordinates from box
    xmin, ymin, xmax, ymax = coords
    # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    # nplate = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
    nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)] ### cropping the number plate from the whole image


    ocr_result = reader.readtext(nplate)



    text = filter_text(region=nplate, ocr_result=ocr_result, region_threshold= region_threshold)

    if len(text) ==1:
        text = text[0].upper()
    return text


### to filter out wrong detections 

def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    
    plate = [] 
    print(ocr_result)
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate





### ---------------------------------------------- Main function -----------------------------------------------------

def main(img_path=None, vid_path=None,vid_out = None):

    print(f"[INFO] Loading model... ")
    ## loading the custom trained model
    #model =  torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt',force_reload=True) ## if you want to download the git repo and then run the detection
    model =  torch.hub.load('./yolov5-master', 'custom', source ='local', path='last.pt',force_reload=True) ### The repo is stored locally

    classes = model.names ### class names in string format




    ### --------------- for detection on image --------------------
    if img_path != None:
        print(f"[INFO] Working with image: {img_path}")
        img_out_name = f"./output/result_{img_path.split('/')[-1]}"

        frame = cv2.imread(img_path) ### reading the image
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        results = detectx(frame, model = model) ### DETECTION HAPPENING HERE    

        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

        frame = plot_boxes(results, frame,classes = classes)
        

        cv2.namedWindow("img_only", cv2.WINDOW_NORMAL) ## creating a free windown to show the result

        while True:
            # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

            cv2.imshow("img_only", frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                print(f"[INFO] Exiting. . . ")

                cv2.imwrite(f"{img_out_name}",frame) ## if you want to save he output result.

                break

    ### --------------- for detection on video --------------------
    elif vid_path !=None:
        print(f"[INFO] Working with video: {vid_path}")

        ## reading the video
        cap = cv2.VideoCapture(vid_path)


        if vid_out: ### creating the video writer if video output path is given

            # by default VideoCapture returns float instead of int
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            print(fps)
            codec = cv2.VideoWriter_fourcc(*'H264') ##(*'XVID')
            out = cv2.VideoWriter(vid_out, codec, 10, (width, height))
        
        # assert cap.isOpened()
        frame_no = 1

        cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)
        while True:
            # start_time = time.time()
            ret, frame = cap.read()
            if ret  and frame_no %1 == 0:
                print(f"[INFO] Working with frame {frame_no} ")

                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                results = detectx(frame, model = model)
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)


                frame = plot_boxes(results, frame,classes = classes)
                
                cv2.imshow("vid_out", frame)
                if vid_out:
                    print(f"[INFO] Saving output video. . . ")
                    out.write(frame)
                #storage.child('video').put('webcam_result.mp4')
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break1(cv2=cv2,out=out)
                    
                    break
                    
                frame_no += 1
        
        print(f"[INFO] Clening up. . . ")
        # ### releaseing the writer
        
        # #storage.child('video').put('webcam_result.mp4')
        
        # # storage.child('video').put('webcam_result.mp4')
        
        # ## closing all windows
        cv2.destroyAllWindows()
        


# def job():
#       today = date.today()
#       print(today)
#       headers = {"Authorization": "Bearer  ya29.A0AVA9y1vE9sOovnTbeulI-KRFRi5ZDJ4zGLkhFDhBGkpbuhGO_o-cXOBz6JsSCf-qTDRx2iDkid_LGFeXl4aupqcIuM8H-sY5g2282y0ux-DUA9bFjfM8Kr6tH_R6k9mXEly1Qf_YvJCtbsyuto7ohIkgUbQUaCgYKATASATASFQE65dr8MTFawxUI5dHnPnVS1hQCbg0163"}
#       para = {
#       "name": str(today)+".jpg",
#       }
#       files = {
#       'data': ('metadata', json.dumps(para), 'application/json; charset=UTF-8'),
#       'file': open(str(today)+".jpg", "rb")
#       }
#       r = requests.post(
#       "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
#       headers=headers,
#       files=files
#       )
#       print(r.text)


# def break1(out):
    
#     print(f"[INFO] Clening up. . . ")

#     out.release()
#     cv2.destroyAllWindows()

### -------------------  calling the main function-------------------------------


#main(vid_path="VID20220807211416.mp4",vid_out="VID.mp4") ### for custom video
#main(vid_path=0,vid_out=str(today)+".mp4")#### for webcam

#main(img_path="./test_images/Cars74.jpg") ## for image
            

# def job():
#       today = date.today()
#       print(today)
#       headers = {"Authorization": "Bearer  ya29.A0AVA9y1vE9sOovnTbeulI-KRFRi5ZDJ4zGLkhFDhBGkpbuhGO_o-cXOBz6JsSCf-qTDRx2iDkid_LGFeXl4aupqcIuM8H-sY5g2282y0ux-DUA9bFjfM8Kr6tH_R6k9mXEly1Qf_YvJCtbsyuto7ohIkgUbQUaCgYKATASATASFQE65dr8MTFawxUI5dHnPnVS1hQCbg0163"}
#       para = {
#       "name": str(today)+".mp4",
#       }
#       files = {
#       'data': ('metadata', json.dumps(para), 'application/json; charset=UTF-8'),
#       'file': open(str(today)+".mp4", "rb")
#       }
#       r = requests.post(
#       "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
#       headers=headers,
#       files=files
#       )
#       print(r.text)

# #schedule.every(10).seconds.do(main(vid_path=0,vid_out=str(today)+".mp4"))
# # schedule.every(10).seconds.do(break1(out=out))
# # schedule.every(10).seconds.do(job)



# while True:
 
#     # Checks whether a scheduled task
#     # is pending to run or not
#     schedule.run_pending()
#     time.sleep(1)

# class mini():
#     def __init__(self):
#         root = Tk()

# #label
#         Label(root,text="HIGHWAY CATTLE DETECTION AND OBSTACLES").pack()
#         Label(root,text="enter the camera code").pack()
#         Button(root,text='submit',command=self.cal).pack()
#         self.c = Entry(root,width=50,borderwidth=5).pack()
#         root.mainloop()

#         #Text Input

#         #self.c = Entry(root,width=50,borderwidth=5).pack()


        
# # root = Tk()

# # #label
# # Label(root,text="HIGHWAY CATTLE DETECTION AND OBSTACLES").pack()
# # Label(root,text="enter the camera code").pack()
# #Text Input

# #c = Entry(root,width=50,borderwidth=5).pack()
# #print(c)
#     def cal(self):
#         print(self.c.get())
#         code = int(self.c.get())
#         main(vid_path=code,vid_out=str(today)+".mp4")

# m=mini()
# #button
# # Button(root,text='submit',command=cal).pack()
# # root.mainloop()

# Import module
def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing. 
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports,non_working_ports

# from tkinter import *

# # Create object
# root = Tk()

# # Adjust size
# root.title("HIGHWAY CATTLE DETECTION AND OBSTACLE")
# root.geometry("700x700" )

# # Change the label text
# def show():
    
# 	main(vid_path=int(clicked.get()),vid_out=None)


# a,w,b=list_ports()

# # Dropdown menu options
# options = w

# # datatype of menu text
# clicked = StringVar()

# # initial menu text
# clicked.set( 0 )

# # Create Dropdown menu
# label = Label( root , text = "NO OF CAMERA PORTS AVAILABE " )
# label.pack()
# drop = OptionMenu( root , clicked , *options )
# drop.pack()

# # Create button, it will change label text
# button = Button( root , text = "click Me" , command = show ).pack()

# Create Label




import tkinter
from tkinter import messagebox


win = tkinter.Tk()
win.title("Registraiion form")
win.geometry('440x440')
win.configure(bg='#333333')

def login(u,p):
    
    
    
    password = database.child(u).get().val()
    
    if p==password:
        messagebox.showinfo(title="Login Success", message="You successfully logged in.")
        root=Toplevel()
        root.configure(bg='#333333')
        root.title("HIGHWAY CATTLE DETECTION AND OBSTACLE")
        root.geometry("700x700" )

        # Change the label text
        def show():
            
            main(vid_path=int(clicked.get()),vid_out=None)


        a,w,b=list_ports()

        # Dropdown menu options
        options = w

        # datatype of menu text
        clicked = StringVar()

        # initial menu text
        clicked.set( 0 )

        # Create Dropdown menu
        label = Label( root , text = "NO OF CAMERA PORTS AVAILABE " )
        label.pack()
        drop = OptionMenu( root , clicked , *options )
        drop.pack()

        # Create button, it will change label text
        button = Button( root , text = "click Me" , command = show ).pack()

    else:
        messagebox.showerror(title="Error", message="Invalid login.")



# Creating widgets
# login_label = tkinter.Label(
#     frame, text="Login", bg='#333333', fg="#FF3399", font=("Arial", 30))
# username_label = tkinter.Label(
#     frame, text="Username", bg='#333333', fg="#FFFFFF", font=("Arial", 16))
# username_entry = tkinter.Entry(frame, font=("Arial", 16))
# password_entry = tkinter.Entry(frame, show="*", font=("Arial", 16))
# password_label = tkinter.Label(
#     frame, text="Password", bg='#333333', fg="#FFFFFF", font=("Arial", 16))
# login_button = tkinter.Button(
#     frame, text="Login", bg="#FF3399", fg="#FFFFFF", font=("Arial", 16), command=login)

# # Placing widgets on the screen
# login_label.grid(row=0, column=0, columnspan=2, sticky="news", pady=40)
# username_label.grid(row=1, column=0)
# username_entry.grid(row=1, column=1, pady=20)
# password_label.grid(row=2, column=0)
# password_entry.grid(row=2, column=1, pady=20)
# login_button.grid(row=3, column=0, columnspan=2, pady=30)

# frame.pack()
def reg():
    print(type(username_e.get()))
    if username_e.get()!="":
        database.child(str(username_e.get())).set(password_e.get())
        print("1")
    top=Toplevel()
    top.title("Login form")
    top.geometry('340x440')
    top.configure(bg='#333333')
    

# Creating widgets
    login_label = tkinter.Label(
        top, text="Login", bg='#333333', fg="#FF3399", font=("Arial", 30))
    username_label = tkinter.Label(
        top, text="Username", bg='#333333', fg="#FFFFFF", font=("Arial", 16))
    username_entry = tkinter.Entry(top, font=("Arial", 16))
    password_entry = tkinter.Entry(top, show="*", font=("Arial", 16))
    password_label = tkinter.Label(
        top, text="Password", bg='#333333', fg="#FFFFFF", font=("Arial", 16))
   

    # Placing widgets on the screen
    login_label.grid(row=0, column=0, columnspan=2, sticky="news", pady=40)
    username_label.grid(row=1, column=0)
    username_entry.grid(row=1, column=1, pady=20)
    password_label.grid(row=2, column=0)
    password_entry.grid(row=2, column=1, pady=20)
    login_button = tkinter.Button(
        top, text="Login", bg="#FF3399", fg="#FFFFFF", font=("Arial", 16),command=lambda:login(username_entry.get(),password_entry.get()))
    login_button.grid(row=3, column=0, columnspan=2, pady=30)
    

    

login_label = tkinter.Label(
        win, text="Signup", bg='#333333', fg="#FF3399", font=("Arial", 30))   
username = tkinter.Label(win , text = "Username", bg='#333333', fg="#FFFFFF", font=("Arial", 16))
username_e = tkinter.Entry(win,font=("Arial", 16))
password=tkinter.Label(win,text="Password", bg='#333333', fg="#FFFFFF", font=("Arial", 16))
password_e=tkinter.Entry(win,font=("Arial", 16),show="*")
button1=tkinter.Button(win,text="Signup", command=reg)
button2=tkinter.Button(win,text="login", command=reg)


login_label.pack()
username.pack()
username_e.pack()
password.pack()
password_e.pack()
button1.pack() 
button2.pack()




win.mainloop()


# Execute tkinter



    