import tkinter as tk
import numpy as np
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import speech_recognition as sr
import openai
import pyttsx3  
openai.api_key = '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' 
engine = pyttsx3.init()  
recognizer = sr.Recognizer()
global res

res=""
def init(res):

    model_response = chat_with_gpt(res)
    engine.say(model_response) 
    engine.runAndWait()
    
    
    
def lung_test():
    global lung
    lung = tk.Tk()
    lung.geometry("1280x720")
    my_font1=("Helvacita", 18, 'bold')
    l1 = tk.Label(lung,text='Add Lung Image',height=10,width=80,font=my_font1)  
    l1.grid(row=1,column=1)
    b1 = tk.Button(lung, text='Upload File',  width=20,command = lambda:upload_file_lung())
    b1.grid(row=2,column=1) 
     

def upload_file_lung():
    global img
    f_types = [('Jpg Files', '*.jpg'),("JPEG Files" ,"*.jpeg"),("PNG images","*.png")]
    filename = filedialog.askopenfilename(filetypes=f_types)
    img = ImageTk.PhotoImage(file=filename)
    img = ImageTk.PhotoImage(Image.open(filename))
    model=load_model('D:/Hack/model_vgg16.h5')
    img=image.load_img(filename,target_size=(224,224))
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    img_data=preprocess_input(x)
    classes=model.predict(img_data,verbose='None')
    lis=classes.tolist()
    if lis==[[0.0,1.0]]:
        res="Pneumonia symptoms"
        print("Pneumonia")
        a=tk.Label(lung,text="Paitent Report\nPneumonia")
        a.grid(columnspan = 3,column=0,row=3)
    else:
        res="Pneumonia symptoms"
        print("Normal")
        a=tk.Label(lung,text="Paitent Report\nNormal")
        a.grid(columnspan = 3,column=0,row=3)
    c1 = tk.Button(lung, text='Clear screen',  width=20, command = lambda:a.config(text=" "))
    c1.grid(row=7,column=1)
    init(res)
    b3 = tk.Button(lung, text='continue with chat',  width=20,command = lambda:chat())
    b3.grid(row=9,column=1)
def skin_cancer():
    global skin
    skin = tk.Tk()
    skin.geometry("1280x720") 
    my_font1=("Helvacita", 18, 'bold')
    l1 = tk.Label(skin,text='Add skin Image',height=10,width=80,font=my_font1)  
    l1.grid(row=1,column=1)
    b1 = tk.Button(skin, text='Upload File',  width=20,command = lambda:upload_file_skin())
    b1.grid(row=2,column=1) 
    
def upload_file_skin():
    global img
    f_types = [('Jpg Files', '*.jpg'),("JPEG Files" ,"*.jpeg"),("PNG images","*.png")]
    filename = filedialog.askopenfilename(filetypes=f_types)
    img = ImageTk.PhotoImage(file=filename)
    img = ImageTk.PhotoImage(Image.open(filename))
    model=load_model('D:/Hack/skin_Cancer20epochs.h5')
    pred_image = image.load_img(filename,target_size=(224, 224))
    pred_image = image.img_to_array(pred_image)
    pred_image = np.expand_dims(pred_image,axis = 0)
    answer = model.predict(pred_image,verbose='None')
    if answer[0][0] < 0.1:
        
        res="BENIGN skin cancer"
        a=tk.Label(skin,text="Paitent Report\nBENIGN skin cancer")
        a.grid(columnspan = 3,column=0,row=3)
    else:
        res="MALIGNANT skin cancer"
        a=tk.Label(skin,text="Paitent Report\nMALIGNANT skin cancer")
        a.grid(columnspan = 3,column=0,row=3)
    c1 = tk.Button(skin, text='Clear screen',  width=20, command = lambda:a.config(text=" "))
    c1.grid(row=7,column=1)
    init(res)
    b3 = tk.Button(skin, text='continue with chat',  width=20,command = lambda:chat())
    b3.grid(row=9,column=1) 





def process_speech_input(audio):
    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        model_response = chat_with_gpt(text)

        print("ChatGPT:", model_response)
        engine.say(model_response) 
        engine.runAndWait()  


    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Error: {0}".format(e))

def chat_with_gpt(message):
    response = openai.Completion.create(
        engine='text-davinci-003', 
        prompt=message,
        max_tokens=50,
        temperature=0.7
    )

    return response.choices[0].text.strip()

def chat():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        process_speech_input(audio)

    






    
wrks = tk.Tk()
wrks.geometry("1280x720")
my_font1=("Helvacita", 18, 'bold')

l1 = tk.Label(wrks,text='choose a test',height=10,width=80,font=my_font1)  
l1.grid(row=1,column=1)
b1 = tk.Button(wrks, text='lung pneumonia',  width=20,command = lambda:lung_test())
b1.grid(row=2,column=1) 
b2 = tk.Button(wrks, text='skin cancer',  width=20,command = lambda:skin_cancer())
b2.grid(row=4,column=1) 
b3 = tk.Button(wrks, text='continue with chat',  width=20,command = lambda:chat())
b3.grid(row=5,column=1) 
wrks.mainloop()


