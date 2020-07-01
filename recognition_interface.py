from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from ttkthemes import themed_tk as tk

import keras
from keras import models
from keras.models import Model
from keras.models import load_model

import os
import time
import numpy as np
import matplotlib.pyplot as plt



class MyApp(Frame):

    def __init__(self, master, *args, **kwargs):

        self.root = master
        self.root.geometry("1000x600")
        self.root.configure(background="#6cb6eb")

        ### BUTTONS ###

        self.load_image_but = Button(self.root, text='Load Image', height=3, width=28, command=self.load_image)
        self.load_image_but.grid(row = 1, column = 1)
        self.load_model_but = Button(self.root, text='Load Model', height=3, width=28, command=self.load_model_)
        self.load_model_but.grid(row = 2, column=1)
        self.prediction_but = Button(self.root, text="Prediction", height=3, width=28, command=self.prediction)
        self.prediction_but.grid(row=4, column=1)
        self.quit_but = Button(self.root, text="Quit", height=3, width=28, command=quit)
        self.quit_but.grid(row=10, column=1)

        self.layer1_but = Button(self.root, text='Filters in layer 1', height=3, width=28, command=lambda: self.display_activation(self.activations, 2, 2, 0))
        self.layer1_but.grid(row=6, column=1)
        self.layer2_but = Button(self.root, text='Filters in layer 2', height=3, width=28, command=lambda: self.display_activation(self.activations, 2, 2, 1))
        self.layer2_but.grid(row=7, column=1)
        self.layer3_but = Button(self.root, text='Filters in layer 3', height=3, width=28, command=lambda: self.display_activation(self.activations, 2, 2, 2))
        self.layer3_but.grid(row=8, column=1)
        self.layer4_but = Button(self.root, text='Filters in layer 4', height=3, width=28, command=lambda: self.display_activation(self.activations, 2, 2, 3))
        self.layer4_but.grid(row=9, column=1)

        ###LABELS###

        self.empty_lab0 = Label(self.root, width=2, background='#6cb6eb')
        self.empty_lab0.grid(row=0, column=0)
        self.pred_lab = Label(self.root, text="", height=2, background='#6cb6eb')
        self.pred_lab.grid(row=5, column=1)
        self.empty_lab2 = Label(self.root, background='#6cb6eb', height=2, text="")
        self.empty_lab2.grid(row=3, column=1)
        self.empty_lab3 = Label(self.root, width=4, background='#6cb6eb')
        self.empty_lab3.grid(row=4, column=2)
        self.title_lab = Label(self.root, background='#6cb6eb', text="")
        self.title_lab.grid(row=1, column=3)
        self.empty_lab4 = Label(self.root, width=2, background='#6cb6eb')
        self.empty_lab4.grid(row=11, column=0)
        self.empty_lab5 = Label(self.root, width=2, background='#6cb6eb')
        self.empty_lab5.grid(row=12, column=0)
        self.path_lab = Label(self.root, background='#6cb6eb', text="")
        self.path_lab.grid(row=11, column=1, columnspan=4, rowspan=2)

    def load_image(self):
        self.x = filedialog.askopenfilename(title='Select the image')
        img = Image.open(self.x)
        img = img.resize((600, 450), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(root, image = img)
        panel.image = img
        panel.grid(row = 2, column=3, rowspan=9)
        img_title = "The " + str(os.path.basename(self.x)) + " image has been loaded."
        img_path = "The image has been loaded from the: " + str(self.x) + "."
        self.title_lab.config(text=img_title, font=("Arial, 11"))
        self.path_lab.config(text=img_path)
        self.load_image_but.config(text="Change Image")

    def load_model_(self):
        start_loading = time.time()
        best_model = 'best_model.h5'
        self.model = load_model(best_model)
        loading_period = str(np.round(time.time() - start_loading, 3))
        loaded = "Model loaded in " + loading_period + " sec."
        self.empty_lab2.config(text=loaded)

    def prediction(self):
        self.label_names = {0: 'cyclone', 1: 'earthquake', 2: 'flood', 3: 'wildfire'}
        image = keras.preprocessing.image.load_img(self.x, color_mode="rgb", target_size=(180, 180))
        input_arr = keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        pred = self.model.predict(input_arr)
        pred_title = self.label_names[np.argmax(pred)]
        pred_title = "The image has been classified as\n" + str(pred_title) + "."
        self.pred_lab.config(text=pred_title)
        layer_outputs = [layer.output for layer in self.model.layers]
        activation_model = Model(inputs=self.model.input, outputs=layer_outputs)
        self.activations = activation_model.predict(input_arr)


    def display_activation(self, activations, row_size, col_size, act_index):
        activation = activations[act_index]
        activation_index=0
        fig, ax = plt.subplots(row_size, col_size, figsize=(5, 4.5))
        fig.patch.set_facecolor("#6cb6eb")
        fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
        for row in range(0, row_size):
            for col in range(0, col_size):
                ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
                ax[row][col].axis('off')
                activation_index += 1
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = ImageTk.PhotoImage(image=Image.fromarray(data))
        panel = Label(root, image = img)
        panel.image = img
        panel.grid(row = 2, column=6, rowspan=9)


if __name__ == "__main__":
    root = Tk()
    App = MyApp(root)
    root.title("Disaster Interface")
    root.mainloop()
