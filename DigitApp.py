import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model
from TrainModel import MnistModelTrainer
import os

class ImageCanvas(tk.Canvas):
    def __init__(self, root):
        super().__init__(root, width=280, height=280, bg="black")
        self.pack(pady=20)
        self.root = root
        self.update_image_obj()

    def update_image_obj(self):
        self.delete("all")
        self.image = Image.new("L", (280, 280), "black")
        self.draw = ImageDraw.Draw(self.image)

    
    def create_line(self, x1, y1, x2, y2, fill="white", width=5):
        super().create_line(x1, y1, x2, y2, fill=fill, width=width)
        self.draw.line([x1, y1, x2, y2], fill=fill, width=width)
        


class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sayi Tanima Uygulamasi")

        self.canvas = ImageCanvas(self.root)

        self.submit_button = tk.Button(self.root, text="Tahmin Et", command=self.predict_digit)
        self.submit_button.pack(pady=10, side="left")
        self.clear_button = tk.Button(self.root, text="Temizle", command=self.clear)
        self.clear_button.pack(pady=10, side="right")

        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        self.last_x = None
        self.last_y = None

        self.drawing = False
        self.model = DigitApp.get_model()


    @staticmethod
    def get_model():
        if not os.path.exists("models/mnist_model.h5"):
            train_model = MnistModelTrainer()
            train_model.load_dataset()
            train_model.create_model()
            train_model.train_model(epochs=5)
            train_model.save_model("models/mnist_model.h5")

        model = load_model("models/mnist_model.h5")
        return model
    
    def start_drawing(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def stop_drawing(self, event):
        self.drawing = False

    def draw(self, event):
        if self.drawing:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, fill="white", width=5)
            self.last_x = event.x
            self.last_y = event.y

    def predict_digit(self):
        img = self.canvas.image.resize((28, 28))

        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28)

        prediction = self.model.predict(img_array)
        digit = np.argmax(prediction[0])
        confidence = prediction[0][digit] * 100

        messagebox.showinfo("Tahmin", f"Tahmin Edilen Rakam: {digit}\nGüven: {confidence:.2f}%")

    def clear(self):
        self.canvas.update_image_obj()



if __name__ == "__main__":
    root = tk.Tk()
    app = DigitApp(root)
    root.mainloop() 
