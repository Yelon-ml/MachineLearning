from tkinter import *
#from tkinter.ttk import *
#import tkinter as tk
from ttkthemes import themed_tk as tk
from tkinter import filedialog
import numpy as np
from PIL import ImageTk, Image
import keras
from keras import models
from keras.models import Model
from keras.models import load_model
import matplotlib.pyplot as plt
import time


root = tk.ThemedTk()
root.get_themes()
root.set_theme("winxpblue")
root.title("Disaster Recognition Interface")
root.geometry("1000x600")
root.configure(background='#6cb6eb')

label_names = {0: 'cyclone', 1: 'earthquake', 2: 'flood', 3: 'wildfire'}


def load_img():
    global x
    x = filedialog.askopenfilename(title='Select the image')
    img = Image.open(x)
    img = img.resize((600, 450), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image = img)
    panel.image = img
    panel.grid(row = 2, column=3, rowspan=9)
    img_title = "Image from the below path has been loaded:\n\n" + str(x) + "."
    title_lab.config(text=img_title)
    return img


def load_model_():
    start_loading = time.time()
    global model
    best_model = 'best_model.h5'
    model = load_model(best_model)
    loading_period = str(np.round(time.time() - start_loading, 3))
    loaded = "Model loaded in " + loading_period + " sec."
    empty_lab2.config(text=loaded)
    print(loading_period)
    return model

def prediction():
    global input_arr, activations
    image = keras.preprocessing.image.load_img(x, color_mode="rgb", target_size=(180, 180))
    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    pred = model.predict(input_arr)
    pred_title = label_names[np.argmax(pred)]
    pred_title = "The image has been classified as\n" + str(pred_title) + "."
    pred_lab.config(text=pred_title)
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(input_arr)
    return input_arr, activations


empty_lab0 = Label(root, width=2, background='#6cb6eb')
empty_lab0.grid(row=0, column=0)
pred_lab = Label(root, text="", height=2, background='#6cb6eb')
pred_lab.grid(row=5, column=1)
empty_lab2 = Label(root, background='#6cb6eb', height=2, text="")
empty_lab2.grid(row=3, column=1)
empty_lab3 = Label(root, background='#6cb6eb')
empty_lab3.grid(row=4, column=2)

title_lab = Label(root,  background='#6cb6eb', text=" ")
title_lab.grid(row=1, column=3)

select_image_but = Button(root, text='Load Image', height=3, width=28, command=load_img)
select_image_but.grid(row = 1, column=1)
load_model_but = Button(root, text='Load Model', height=3, width=28, command=load_model_)
load_model_but.grid(row = 2, column=1)


def display_activation(activations, col_size, act_index):
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(1, col_size, figsize=(10, 4))
    for col in range(0, col_size):
        ax[col].imshow(activation[0, :, :, activation_index], cmap='gray')
        ax[col].axis('off')
        activation_index += 1
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = ImageTk.PhotoImage(image=Image.fromarray(data))
    panel = Label(root, image = img)
    panel.image = img
    panel.grid(row = 2, column=3, rowspan=9)
    plt.show()


layer1_but = Button(root, text='Filters in layer 1', height=3, width=28, command=lambda: display_activation(activations, 4, 0))
layer1_but.grid(row=6, column=1)
layer2_but = Button(root, text='Filters in layer 2', height=3, width=28, command=lambda: display_activation(activations, 4, 1))
layer2_but.grid(row=7, column=1)
layer3_but = Button(root, text='Filters in layer 3', height=3, width=28, command=lambda: display_activation(activations, 4, 2))
layer3_but.grid(row=8, column=1)
layer4_but = Button(root, text='Filters in layer 4', height=3, width=28, command=lambda: display_activation(activations, 4, 3))
layer4_but.grid(row=9, column=1)

but_pred = Button(root, text="Prediction", height=3, width=28, command=prediction)
but_pred.grid(row=4, column=1)

but_quit = Button(root, text="Quit", height=3, width=28, command=quit)
but_quit.grid(row=10, column=1)

root.mainloop()
