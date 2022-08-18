'''
CODER ZERO
connect with me at: https://www.youtube.com/channel/UCKipQAvBc7CWZaPib4y8Ajg
How to train custom yolov5: https://youtu.be/12UoOlsRwh8
DATASET: 1) https://www.kaggle.com/datasets/deepakat002/indian-vehicle-number-plate-yolo-annotation
         2) https://www.kaggle.com/datasets/elysian01/car-number-plate-detection
'''
### importing required libraries
from asyncio.windows_events import NULL
import torch
import cv2
import time
#import pyrebase
# import pytesseract
import re
import numpy as np
#import easyocr
import pandas as pd
from playsound import playsound
#import geocoder
#from openpyxl import load_workbook
##### DEFINING GLOBAL VARIABLE
# EASY_OCR = easyocr.Reader(['en']) ### initiating easyocr
# OCR_TH = 0.2


# config={  
#   'apiKey': "AIzaSyCLscXyqA0KJCsUnebYEqn6TXLlA_HTc1g",
#   'authDomain': "sampleproject-ef4d0.firebaseapp.com",
#   'databaseURL': "https://sampleproject-ef4d0-default-rtdb.firebaseio.com",
#   'projectId': "sampleproject-ef4d0",
#   'storageBucket': "sampleproject-ef4d0.appspot.com",
#   'messagingSenderId': "863988637736",
#   'appId': "1:863988637736:web:25674b1d514d62e91b951d",
#   'measurementId': "G-ZL0TYCYTXP"
# }
# firebase=pyrebase.initialize_app(config)
# storage = firebase.storage()
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
    model =  torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5x6.pt',force_reload=True) ## if you want to download the git repo and then run the detection
    #model =  torch.hub.load('./yolov5-master', 'custom', source ='local', path='yolov5x6.pt',force_reload=True) ### The repo is stored locally

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
            codec = cv2.VideoWriter_fourcc(*'mp4v') ##(*'XVID')
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
                    break
                frame_no += 1
        
        print(f"[INFO] Clening up. . . ")
        ### releaseing the writer
        #storage.child('video').put('webcam_result.mp4')
        out.release()
        # storage.child('video').put('webcam_result.mp4')
        
        ## closing all windows
        cv2.destroyAllWindows()



### -------------------  calling the main function-------------------------------


#main(vid_path="VID20220807211416.mp4",vid_out="VID.mp4") ### for custom video
main(vid_path=0,vid_out="webcam_result.mp4")#### for webcam

#main(img_path="./test_images/Cars74.jpg") ## for image
            

