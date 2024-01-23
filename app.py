import tkinter as tk
from tkinter import Canvas, Button
from PIL import Image, ImageDraw
import numpy as np
from tensorflow import keras

# Load the MNIST dataset
mnist = keras.datasets.mnist
(_, _), (test_images, _) = mnist.load_data()
test_images = test_images / 255.0

# Load the trained model
model = keras.models.load_model('G:\\your\\model\\path\\mnist_model.h5')

# Create a Tkinter window
window = tk.Tk()
window.title("Digit Recognizer")

# Create a canvas for drawing
canvas = Canvas(window, width=280, height=280, bg='white')
canvas.grid(row=0, column=0, padx=10, pady=10)

# Create an image to draw on
image = Image.new("L", (280, 280), 0)
draw = ImageDraw.Draw(image)

# Function to recognize the drawn digit
def recognize_digit():
    # Resize the drawn image to 28x28
    digit_image = image.resize((28, 28))

    # Convert the image to a numpy array
    digit_array = np.array(digit_image)

    # Normalize the pixel values
    digit_array = digit_array / 255.0

    # Reshape the array to match the model input shape
    digit_array = np.reshape(digit_array, (1, 28, 28))

    # Make a prediction using the model
    prediction = model.predict(digit_array)

    # Get the predicted digit
    predicted_digit = np.argmax(prediction)

    # Display the predicted digit
    result_label.config(text=f"Predicted Digit: {predicted_digit}")

# Function to clear the canvas
def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, 280, 280), fill="white")

# Button to recognize the drawn digit
recognize_button = Button(window, text="Recognize Digit", command=recognize_digit)
recognize_button.grid(row=1, column=0, pady=10)

# Button to clear the canvas
clear_button = Button(window, text="Clear Canvas", command=clear_canvas)
clear_button.grid(row=2, column=0, pady=10)

# Label to display the predicted digit
result_label = tk.Label(window, text="")
result_label.grid(row=3, column=0, pady=10)

# Function to handle mouse movements
def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=8)
    draw.line([x1, y1, x2, y2], fill="black", width=16)

# Bind the paint function to mouse movements
canvas.bind("<B1-Motion>", paint)

# Run the Tkinter event loop
window.mainloop()
