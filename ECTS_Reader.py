# -*- coding: utf-8 -*-
"""
Created on Tue May 11 21:02:18 2021

@author: W10
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 10 02:38:18 2021

@author: W10
"""

import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageTk
from pandas import DataFrame
import os 
from tkinter import filedialog
import tkinter as tk 
from tkinter.ttk import Combobox
import pandas as pd
from fpdf import FPDF
import os
import shutil
import matplotlib.pyplot as plt
import sqlite3
import tkinter.font as tkFont
from tkinter_custom_button import TkinterCustomButton
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\\tesseract.exe'


def reportscreen():
    
#    windowtwo.destroy()
    windowthree = tk.Toplevel(windowtwo)
    windowthree.configure(bg='white')
    windowthree.geometry("700x300+300+100")
#    windowthree.title("European Credit Transfer System (ECTS) Workload Form Reader ")
    
    
    connection = sqlite3.connect("ectsformreader.db")
    cursor = connection.cursor()
    
    
    classname_ddm = pd.read_sql_query("SELECT DISTINCT Class FROM results_omr", connection)
    classname_ddm_list= classname_ddm.values.tolist()

    fontStyle = tkFont.Font(size=12)
    
    
    introduction= tk.Label(windowthree,text="    This screen creates evaluation report for classes     ",
                           bg="white",font=tkFont.Font(size=13, weight=tkFont.BOLD),bd=2,relief=tk.SOLID)
    introduction.place(x=120,y=20,height=40)
    
    
    header4= tk.Label(windowthree,text="Choose the Class You Want to Get Report :",
                      bg="white",font=fontStyle,bd=4)
    header4.place(x=20,y=115)     
    
    from matplotlib import rcParams
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    PLOT_DIR = 'plots'
    def construct(df):
        # Delete folder if exists and create it again
        try:
            shutil.rmtree(PLOT_DIR)
            os.mkdir(PLOT_DIR)
        except FileNotFoundError:
            os.mkdir(PLOT_DIR)        
        # Iterate over all months in 2020 except January
        for i in range(len(df)):        
            #plot(data=generate_sales_data(month=i), filename=f'{PLOT_DIR}/{i}.png')
            df.iloc[i].T.plot.pie(figsize=(0.25,0.25),fontsize=1,legend = False)
            plt.title(f'Question of {i}', fontsize=4)
            filename= (f'{PLOT_DIR}/{i}.png')
            plt.savefig(filename, dpi=750, bbox_inches='tight', pad_inches=0)
            plt.close()
        # Construct data shown in document
        counter = 0
        pages_data = []
        temp = []
        # Get all plots
        files = os.listdir(PLOT_DIR)
        # Sort them by month - a bit tricky because the file names are strings
        # Iterate over all created visualization
        for fname in files:
            # We want 3 per page
            if counter == 2:
                pages_data.append(temp)
                temp = []
                counter = 0
            temp.append(f'{PLOT_DIR}/{fname}')
            counter += 1
        return [*pages_data, temp]
    
    
    
    class PDF(FPDF):
        def __init__(self):
            super().__init__()
            self.WIDTH = 210
            self.HEIGHT = 297
            
        def header(self):
            # Custom logo and positioning
            # Create an `assets` folder and put any wide and short image inside
            # Name the image `logo.png`
    #        self.image('logo.png', 10, 8, 20)
            self.set_font('Arial', 'B', 11)
            self.cell(self.WIDTH - 80)
            self.cell(60, 1, 'Survey Report', 0, 0, 'R')
            self.ln(20)
            
        def footer(self):
            # Page numbers in the footer
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128)
            self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')
    
        def page_body(self, images):
            # Determine how many plots there are per page and set positions
            # and margins accordingly
            if len(images) == 3:
                self.image(images[0], 15, 50, self.WIDTH - 30)
                self.image(images[1], 15, self.WIDTH / 2 + 50, self.WIDTH - 30)
                self.image(images[2], 15, self.WIDTH / 2 + 110, self.WIDTH - 30)
            elif len(images) == 2:
                self.image(images[0], 15, 25, self.WIDTH - 30)
                self.image(images[1], 15, self.WIDTH / 2 + 50, self.WIDTH - 30)
            else:
                self.image(images[0], 15, 25, self.WIDTH - 30)
                
        def print_page(self, images):
            # Generates the report
            self.add_page()
            self.page_body(images)
    pdf = PDF()
    
    
    def create_report():
       
        results_new=selected_df.fillna(0)
        new_data = results_new.ix[:,1:]
        print(new_data)
        all_freq = {}
        all_dict=[]
        a=0     
        for j in range(len(new_data.columns)):
            
            all_freq = {}
            for i in new_data.loc[a]:
                if i in all_freq:
                    all_freq[i] += 1            
                else:
                    all_freq[i] = 1
            all_dict.append(all_freq)  
    #    print ("Count of all characters in column is :\n "
    #                                        +  str(all_dict))    
        
        all_df= pd.DataFrame.from_dict(all_dict)
        plots_per_page = construct(all_df)
        for elem in plots_per_page:
            pdf.print_page(elem)    
        pdf.output('ReportV1.pdf', 'F')  
        
        done= tk.Label(windowthree,text="    Download Successfully Done!     ",
                               bg="white",font=tkFont.Font(size=9, weight=tkFont.BOLD))
        done.place(x=500,y=220,height=40)   
        return 
    
    
    
    def combobox_selection(event):
        
        global selected_df
        selected_classname=dropdownmenu.get()
        selected_df = pd.read_sql_query("SELECT * FROM results_omr WHERE Class = '{}'".format(selected_classname), connection)
        dowloadreport_button=TkinterCustomButton(master=windowthree,text="Download Report of the {} Class".format(selected_classname),
                                                 corner_radius=20,fg_color="OliveDrab2",
                                                 text_color="black",text_font=tkFont.Font(size=10, weight=tkFont.BOLD),
                                                 command=create_report,hover_color="#2bbf1d",width=300)
        dowloadreport_button.place(x=200,y=220)    
        return selected_df   
    
    z2=tk.StringVar()
    dropdownmenu= Combobox(windowthree,values=classname_ddm_list,textvariable=z2)
    dropdownmenu['state'] = 'readonly'
    dropdownmenu.pack()
    dropdownmenu.place(x=350,y=120)
    
    
    dropdownmenu.bind('<<ComboboxSelected>>', combobox_selection)    
    
    windowthree.resizable(False,False)
    
    windowthree.mainloop()    
    




def Upload_orginal_image(event=None):
    global filename
    filename = filedialog.askopenfilename()
    return filename

def getpath_orgimg():
    directory = filename
    global imgQ, h, w
    imgQ = cv2.imread(directory)
    h,w,c=imgQ.shape
    imgQ =cv2.resize (imgQ, (w,h))
    return imgQ, h, w

roi_ocr= []
def OCRboxdetect():
    imgQ, h, w = getpath_orgimg()
    from boxdetect import config
    from boxdetect.pipelines import get_boxes
    from boxdetect.pipelines import get_checkboxes
    config = config.PipelinesConfig()
    
    config.width_range = (35,175)
    config.height_range = (30,90)
    config.scaling_factors = [
    0.9,0.85,0.99,0.99,0.98,0.97,0.89,0.6,0.5,0.4,0.78,0.80,0.3,0.2,0.1,0.33,0.44,0.22,0.11]
    config.wh_ratio_range = (0.6, 5) 
    config.group_size_range = (2, 100)
    
    rects, grouped_rects, org_image, output_image = get_boxes(imgQ, cfg=config, plot=False)
    checkboxes = get_checkboxes(
    imgQ, cfg=config, px_threshold=0.1, plot=False, verbose=False)
    
    coord_list=[]
    for checkbox in checkboxes:
        coord_list.append(checkbox[0])
    coord_list = np.array(coord_list)
    x1 = coord_list[:,0]
    y1= coord_list[:,1]
    x2 = x1 + coord_list[:,2]
    y2 = y1 + coord_list[:,3]
    for i in range(len(coord_list)):
        roi_ocr.append([(x1[i],y1[i]),(x2[i],y2[i])])
    return roi_ocr

roi_omr= []
def OMRboxdetect():
    imgQ, h, w = getpath_orgimg()
    from boxdetect import config
    from boxdetect.pipelines import get_boxes
    from boxdetect.pipelines import get_checkboxes
    config = config.PipelinesConfig()
    
    config.width_range = (35,175)
    config.height_range = (19,25)
    config.scaling_factors = [
    0.9,0.85,0.99,0.99,0.98,0.97,0.89,0.6,0.5,0.4,0.78,0.80,0.3,0.2,0.1,0.33,0.44,0.22,0.11]
    config.wh_ratio_range = (0.6, 5) 
    config.group_size_range = (2, 100)
    
    rects2, grouped_rects2, org_image2, output_image2 = get_boxes(imgQ, cfg=config, plot=False)
    checkboxes2 = get_checkboxes(
    imgQ, cfg=config, px_threshold=0.1, plot=False, verbose=False)
    
    coord_list_omr=[]
    for checkbox2 in checkboxes2:
        coord_list_omr.append(checkbox2[0])
    coord_list_omr = np.array(coord_list_omr)
    x11 = coord_list_omr[:,0]
    y11= coord_list_omr[:,1]
    x22 = x11 + coord_list_omr[:,2]
    y22 = y11 + coord_list_omr[:,3]
    for i in range(len(coord_list_omr)):
        roi_omr.append([(x11[i],y11[i]),(x22[i],y22[i])])
    return coord_list_omr, x11, x22, roi_omr

nop=[]
def getnumberofoptions():   
    global coord_list_omr, x11, x22, roi_omr
    coord_list_omr, x11, x22, roi_omr = OMRboxdetect() 
    for i in range(len(coord_list_omr)):
        if (x22[i]-x11[i]) >= 78:
            number_of_options=5
        elif 63 <= (x22[i]-x11[i]) <= 77:
            number_of_options=4
        elif 48 <= (x22[i]-x11[i]) <= 62:
            number_of_options=3
        elif 19 <= (x22[i]-x11[i]) <= 47:
            number_of_options=2
        nop.append(number_of_options)
    return nop

def Upload_folder():
    global folderpath
    folderpath = filedialog.askdirectory()
    return folderpath

def omr_ocr():
#    path=folderpath
    nop=getnumberofoptions()
    roi_ocr = OCRboxdetect()
    imgQ, h, w = getpath_orgimg()
    global tablo_omr, tablo_ocr
    tablo_omr=DataFrame()
    tablo_ocr=DataFrame()
    myPicList=os.listdir(folderpath)
    
    for j,y in enumerate(myPicList):
        orb= cv2.ORB_create(5000) # features number=1000
        keypoint1, descriptor1 = orb.detectAndCompute(imgQ,None)    
        img =cv2.imread(folderpath +"/" +y)     
        kp2, des2=orb.detectAndCompute(img,None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des2, descriptor1)
        matches.sort(key=lambda x: x.distance)
        good = matches [:int(len(matches)*(25/100))]
        imgMatch= cv2.drawMatches(img,kp2,imgQ,keypoint1,good,None,flags=2)
        h2,w2,c2=imgMatch.shape
        imgMatch =cv2.resize (imgMatch, (w2//2,h2//2))
    
        srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dstPoints = np.float32([keypoint1[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
        imgScan = cv2.warpPerspective(img,M,(w,h))
       
    
     
    #XXXXXXXXXXXXXXXXXXXXXXXXXX Optical Mark Recognition (OMR)  XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        myData_omr=[]
        
        for x,r in enumerate(roi_omr):    
            
            imgCrop= imgScan [r[0][1]:r[1][1],r[0][0]:r[1][0]]
            imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
            imgThresh = cv2.threshold(imgGray, 100, 255,cv2.THRESH_BINARY_INV )[1]    
            
            if (len(imgThresh[0]) % nop[x]) != 0:
                mod=len(imgThresh[0]) % nop[x]
                rng=nop[x]-mod
                for j in range(rng):
                    imgThresh=np.c_[ imgThresh, np.zeros(len(imgThresh))]    
            else:
                imgThresh=imgThresh    
                
            cols = np.hsplit(imgThresh,nop[x])
            boxes=[]
            for box in cols:
                boxes.append(box)
                
            countR=0
            countC=0
            myPixelVal = np.zeros((1,nop[x]))
            
            for image in boxes:    
                totalPixels = cv2.countNonZero(image)
                myPixelVal[countR][countC]= totalPixels
                countC += 1
                if (countC==nop[x]):countR +=1;countC=0
                print(myPixelVal)
    
            options={0:"A",1:"B",2:"C",3:"D",4:"E"}    
            myIndex=[]
            arr = myPixelVal
            
            if np.max(arr) <= 77:
                arr=np.where(arr, "", arr)
            else:
                myIndexVal = np.where(arr == np.amax(arr))
                myIndex.append(options[myIndexVal[1][0]])  
                
                     
            myData_omr.append(myIndex)  
            myData_omr_df = DataFrame(myData_omr)
            myData_omr_df_transpose=myData_omr_df.T    
        tablo_omr=tablo_omr.append(myData_omr_df_transpose)
    
    
    #XXXXXXXXXXXXXXXXXXXXXXXXXX Optical Character Recognition (OCR)  XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        
        myData_ocr=[]
    
        for x,r in enumerate(roi_ocr):
    
            imgCrop= imgScan [r[0][1]:r[1][1],r[0][0]:r[1][0]]
    
            s=pytesseract.image_to_string(imgCrop,lang='tur+eng')
#            print('{} '.format(s))            
            myData_ocr.append(s.strip()) 
            myData_df = DataFrame(myData_ocr)
            myData_df_transpose=myData_df.T
        tablo_ocr=tablo_ocr.append(myData_df_transpose)
    return tablo_omr,tablo_ocr

def get_classname():
    global classname
    classname = class_entry.get("1.0","end-1c")
    class_entry.config(bg="gray60")
    return classname

def createDB():
    tablo_omr, tablo_ocr =omr_ocr()
#    classname=get_classname()
    classname_list=[]
    for i in range(len(tablo_omr)):
        classname_list.append(classname)
    classname_df = DataFrame(classname_list,columns=['Class'])
        
    import sqlite3
    
    connection = sqlite3.connect("ectsformreader.db")
    cursor = connection.cursor()
    
    tablo_omr = classname_df.join(tablo_omr,how='right',lsuffix='', rsuffix='')
    tablo_ocr = classname_df.join(tablo_ocr,how='right',lsuffix='', rsuffix='')
    
    tablo_omr.to_sql("results_omr",con=connection,if_exists='append', index=False, 
                     index_label=None,chunksize=None)
    tablo_ocr.to_sql("results_ocr",con=connection,if_exists='append', index=False, 
                     index_label=None,chunksize=None)
    

    cursor.execute("SELECT * FROM results_omr").fetchall()
    cursor.execute("SELECT * FROM results_ocr").fetchall()

    
    #cursor.close()
    #connection.close()





windowtwo=tk.Tk()
windowtwo.configure(bg='white')
windowtwo.geometry("700x450+300+100")
windowtwo.title("European Credit Transfer System (ECTS) Workload Form Reader ")



exam=tk.PhotoImage(file =("C://Users//W10//Desktop//interface design part1//exam.png"))
windowtwo.iconphoto(False, exam)

fontStyle = tkFont.Font(size=12)

intro= tk.Label(windowtwo,text="    This screen takes the images of the forms to analyze.     ",
                 bg="white",font=tkFont.Font(size=13, weight=tkFont.BOLD),bd=2,relief=tk.SOLID)
intro.place(x=120,y=20,height=40)


header1= tk.Label(windowtwo,text="Please Upload the Original Survey Form",
                 bg="white",font=fontStyle,bd=4)
header1.place(x=35,y=180)

header2= tk.Label(windowtwo,text="Please Upload the Folder of the Survey Forms",
                 bg="white",font=fontStyle,bd=4)
header2.place(x=360,y=180)

header3= tk.Label(windowtwo,text="Enter the name of the Class to be Analyzed",
                 bg="white",font=fontStyle,bd=4)
header3.place(x=20,y=115) 

explanation= tk.Label(windowtwo,text='You need to select the \n downloaded file from the "create" screen',
                 bg="white",font=tkFont.Font(size=9),bd=4)
explanation.place(x=50,y=215) 

explanation2= tk.Label(windowtwo,text='You need to select the \n folder with filled survey form photos',
                 bg="white",font=tkFont.Font(size=9),bd=4)
explanation2.place(x=425,y=215) 


class_entry= tk.Text(windowtwo,width=25,height=2,bd=2,bg="floralwhite")
class_entry.pack()
class_entry.place(x=360,y=110)




upload_button=TkinterCustomButton(master=windowtwo,text="UPLOAD",corner_radius=20,fg_color="OliveDrab2",
                                  text_color="black",text_font=tkFont.Font(size=10, weight=tkFont.BOLD),
                                  command=Upload_orginal_image,hover_color="#2bbf1d")
upload_button.place( x=105,y=270)

upload_button2=TkinterCustomButton(master=windowtwo,text="UPLOAD",corner_radius=20,fg_color="OliveDrab2",
                         text_color="black",text_font=tkFont.Font(size=10, weight=tkFont.BOLD),
                         command=Upload_folder,hover_color="#2bbf1d")
upload_button2.place(x=470,y=270)

evaluate_button=TkinterCustomButton(master=windowtwo,text="EVALUATE!",corner_radius=20,fg_color="SkyBlue2",
                        text_color="black",text_font=tkFont.Font(size=10, weight=tkFont.BOLD),
                        command=createDB,hover_color="#009687")
evaluate_button.place(x=290,y=340)

ok_button=TkinterCustomButton(master=windowtwo,text=" OK!",corner_radius=20,fg_color="palevioletred1",
                        text_color="black",text_font=tkFont.Font(size=10, weight=tkFont.BOLD),
                        command=lambda: get_classname(),width=80,hover_color="#d66779")
ok_button.place(x=600,y=110)

report_button=TkinterCustomButton(master=windowtwo,corner_radius=10,text="   GO TO THE REPORTS  ",relief=tk.RAISED,bd=2,
                        text_font=tkFont.Font(size=10, weight=tkFont.BOLD),width=180,
                        fg_color="gray5",text_color="white",command=reportscreen,hover_color="#242222")
report_button.place(x=500,y=400)




windowtwo.resizable(False,False)

windowtwo.mainloop()



















