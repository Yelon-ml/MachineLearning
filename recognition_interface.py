from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog

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
        self.root.geometry("250x600")
        self.root.configure(background="#6cb6eb")

        ### BUTTONS ###

        self.load_image_but = Button(self.root, text='Load Image', height=3, width=28, command=self.load_image)
        self.load_image_but.grid(row = 1, column = 1)
        self.load_model_but = Button(self.root, text='Load Model', height=3, width=28, command=self.load_model_)
        self.load_model_but.grid(row = 2, column=1)
        self.prediction_but = Button(self.root, text="Prediction", height=3, width=28, command=self.prediction, state="disabled")
        self.prediction_but.grid(row=4, column=1)
        self.quit_but = Button(self.root, text="Quit", height=3, width=28, command=quit)
        self.quit_but.grid(row=10, column=1)

        self.layer1_but = Button(self.root, text='Filters in layer 1', height=3, width=28, command=lambda: self.display_activation(self.activations, 0), state="disabled")
        self.layer1_but.grid(row=6, column=1)
        self.layer2_but = Button(self.root, text='Filters in layer 2', height=3, width=28, command=lambda: self.display_activation(self.activations, 1), state="disabled")
        self.layer2_but.grid(row=7, column=1)
        self.layer3_but = Button(self.root, text='Filters in layer 3', height=3, width=28, command=lambda: self.display_activation(self.activations, 2), state="disabled")
        self.layer3_but.grid(row=8, column=1)
        self.layer4_but = Button(self.root, text='Filters in layer 4', height=3, width=28, command=lambda: self.display_activation(self.activations, 3), state="disabled")
        self.layer4_but.grid(row=9, column=1)

        self.lef_but = Button(self.root, text="<", height=3, width=8, state='disable', command=self.move_in_the_left)
        self.lef_but.grid(row=1, column=5)
        self.right_but = Button(self.root, text=">", height=3, width=8, command=self.move_in_the_right)
        self.right_but.grid(row=1, column=7)

        ### LABELS ###

        self.empty_lab0 = Label(self.root, width=2, background='#6cb6eb')
        self.empty_lab0.grid(row=0, column=0)
        self.pred_lab = Label(self.root, text="", height=2, background='#6cb6eb')
        self.pred_lab.grid(row=5, column=1)
        self.empty_lab2 = Label(self.root, background='#6cb6eb', height=2, text="")
        self.empty_lab2.grid(row=3, column=1)
        self.empty_lab3 = Label(self.root, width=3, background='#6cb6eb')
        self.empty_lab3.grid(row=4, column=2)
        self.title_lab = Label(self.root, width=66, background='#6cb6eb', text="")
        self.title_lab.grid(row=1, column=3)
        self.empty_lab4 = Label(self.root, width=2, background='#6cb6eb')
        self.empty_lab4.grid(row=11, column=0)
        self.empty_lab5 = Label(self.root, width=2, background='#6cb6eb')
        self.empty_lab5.grid(row=12, column=0)
        self.empty_lab6 = Label(self.root, width=2, background='#6cb6eb')
        self.empty_lab6.grid(row=1, column=4)
        self.path_lab = Label(self.root, background='#6cb6eb', text="")
        self.path_lab.grid(row=11, column=1, columnspan=3, rowspan=2)
        self.lab_between_lr = Label(self.root, width=44 , background='#6cb6eb')
        self.lab_between_lr.grid(row=1, column=6)
        self.filter_lab = Label(self.root, background='#6cb6eb', text="")
        self.filter_lab.grid(row=1, column=6)


        ### FUNCTIONALITY ###

    def load_image(self):
        self.x = filedialog.askopenfilename(title='Select the image')
        img = Image.open(self.x)
        img = img.resize((600, 450), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(root, image = img)
        panel.image = img
        panel.grid(row = 2, column=3, rowspan=9, columnspan=1)
        img_title = "The " + str(os.path.basename(self.x)) + " image has been loaded."
        img_path = "The image has been loaded from the: " + str(os.path.dirname(self.x)) + "."
        self.title_lab.config(text=img_title, font=("Arial, 11"))
        self.path_lab.config(text=img_path)
        self.load_image_but.config(text="Change Image")
        self.root.geometry("875x600")
        self.layer1_but.config(state='disable')
        self.layer2_but.config(state='disable')
        self.layer3_but.config(state='disable')
        self.layer4_but.config(state='disable')

    def load_model_(self):
        start_loading = time.time()
        best_model = 'best_model.h5'
        self.model = load_model(best_model)
        loading_period = str(np.round(time.time() - start_loading, 3))
        loaded = "Model loaded in " + loading_period + " sec."
        self.empty_lab2.config(text=loaded)
        self.prediction_but.config(state='normal')

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
        self.layer1_but.config(state='normal')
        self.layer2_but.config(state='normal')
        self.layer3_but.config(state='normal')
        self.layer4_but.config(state='normal')

    def display_activation(self, activations, act_index):
        self.act_index = act_index
        self.activation = activations[self.act_index]
        self.activation_index=0
        fig = plt.figure(figsize=(4.5, 4.5))
        fig.patch.set_facecolor("#6cb6eb")
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.imshow(self.activation[0, :, :, self.activation_index], cmap='gray', interpolation='nearest')
        plt.axis('off')
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = ImageTk.PhotoImage(image=Image.fromarray(data))
        panel = Label(root, image = img)
        panel.image = img
        panel.grid(row = 2, column=5, rowspan=9, columnspan=3)
        filter_title = "Filter " + str(self.activation_index + 1) + " of " + str(self.activation[0, :, :].shape[2]) + " in layer " + str(self.act_index + 1)
        self.filter_lab.config(text=filter_title)
        self.root.geometry("1355x600")
        if self.activation_index == 0:
            self.lef_but.config(state='disabled')

    def move_in_the_right(self):
        self.activation_index+=1
        fig = plt.figure(figsize=(4.5, 4.5))
        fig.patch.set_facecolor("#6cb6eb")
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.imshow(self.activation[0, :, :, self.activation_index], cmap='gray')
        plt.axis('off')
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = ImageTk.PhotoImage(image=Image.fromarray(data))
        panel = Label(root, image = img)
        panel.image = img
        panel.grid(row = 2, column=5, rowspan=9, columnspan=3)
        filter_title = "Filter " + str(self.activation_index + 1) + " of " + str(self.activation[0, :, :].shape[2]) + " in layer " + str(self.act_index + 1)
        self.filter_lab.config(text=filter_title)
        self.lef_but.config(state='normal')
        if self.activation_index == int(self.activation[0, :, :].shape[2] - 1):
            self.right_but.config(state='disabled')

    def move_in_the_left(self):
        self.activation_index-=1
        fig = plt.figure(figsize=(4.5, 4.5))
        fig.patch.set_facecolor("#6cb6eb")
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.imshow(self.activation[0, :, :, self.activation_index], cmap='gray')
        plt.axis('off')
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = ImageTk.PhotoImage(image=Image.fromarray(data))
        panel = Label(root, image = img)
        panel.image = img
        panel.grid(row = 2, column=5, rowspan=9, columnspan=3)
        filter_title = "Filter " + str(self.activation_index + 1) + " of " + str(self.activation[0, :, :].shape[2]) + " in layer " + str(self.act_index + 1)
        self.filter_lab.config(text=filter_title)
        self.right_but.config(state='normal')
        if self.activation_index == 0:
            self.lef_but.config(state='disabled')


if __name__ == "__main__":
    root = Tk()
    App = MyApp(root)
    root.title("Disaster Interface")
    root.mainloop()
