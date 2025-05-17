import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import tkinter as tk
from tkinter import Button, Canvas, Label, Frame
import serial
import serial.tools.list_ports
import time
import os

# Arduino serial setup
arduino = None
serial_port = 'COM3'  # Update this for your system
try:
    print(f"Connecting to {serial_port}")
    arduino = serial.Serial(serial_port, 9600, timeout=1)
    time.sleep(2)  # Wait for Arduino reset
    print(f"Connected to Arduino on {serial_port}")
except serial.SerialException as e:
    print(f"Serial error: {e}")
    print("Available ports:")
    for port in serial.tools.list_ports.comports():
        print(f" - {port.device}")
    print("Continuing without Arduino.")

# Define the CNN model - IMPORTANT: This now matches the training architecture exactly
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # [1, 28, 28] → [16, 26, 26]
        x = F.max_pool2d(x, 2)               # → [16, 13, 13]
        x = F.relu(self.bn2(self.conv2(x)))  # → [32, 11, 11]
        x = F.max_pool2d(x, 2)               # → [32, 5, 5]
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the trained model
try:
    model = DigitCNN()
    model.load_state_dict(torch.load("digit_cnn.pth", map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

class DigitDrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")
        self.root.geometry("600x650")

        # Drawing canvas
        self.canvas_frame = Frame(root)
        self.canvas_frame.pack(pady=10)
        
        self.canvas = Canvas(self.canvas_frame, width=280, height=280, bg="black")
        self.canvas.pack()
        
        # Bind mouse events for drawing
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset_pos)
        self.old_x = None
        self.old_y = None
        
        # Control buttons
        self.button_frame = Frame(root)
        self.button_frame.pack(pady=10)
        
        self.clear_button = Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=10)
        
        self.predict_button = Button(self.button_frame, text="Predict", command=self.predict)
        self.predict_button.pack(side=tk.LEFT, padx=10)
        
        # Result display
        self.result_frame = Frame(root)
        self.result_frame.pack(pady=10)
        
        self.result_label = Label(self.result_frame, text="Draw a digit (0-9) and click Predict",
                                 font=("Helvetica", 16))
        self.result_label.pack()
        
        # Processed image preview
        self.preview_frame = Frame(root)
        self.preview_frame.pack(pady=10)
        
        self.preview_label = Label(self.preview_frame, text="MNIST Format (28x28)")
        self.preview_label.pack()
        
        self.preview_canvas = Canvas(self.preview_frame, width=140, height=140, bg="black", highlightthickness=1)
        self.preview_canvas.pack()
        
        # Status bar
        self.status_bar = Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Arduino status
        arduino_status = "Connected to Arduino" if arduino else "No Arduino connection"
        self.arduino_label = Label(root, text=arduino_status)
        self.arduino_label.pack(pady=5)

    def draw(self, event):
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, 
                                   width=15, fill="white", capstyle=tk.ROUND, smooth=tk.TRUE)
        self.old_x = event.x
        self.old_y = event.y

    def reset_pos(self, event):
        self.old_x = None
        self.old_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.result_label.config(text="Draw a digit (0-9) and click Predict")
        self.status_bar.config(text="Canvas cleared")

    def predict(self):
        # Get the canvas image
        self.status_bar.config(text="Processing...")
        self.root.update()
        
        # Create a new image from the canvas
        img = Image.new("RGB", (self.canvas.winfo_width(), self.canvas.winfo_height()), color="black")
        drawer = ImageDraw.Draw(img)
        
        # Get all canvas objects and redraw them onto the PIL image
        # This avoids screenshot issues and works on all platforms
        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            if len(coords) >= 4:  # It's a line
                drawer.line(coords, fill="white", width=15)
        
        # Convert to grayscale
        img = img.convert("L")
        
        # MNIST Preprocessing
        # 1. Crop to content with padding
        bbox = self.get_bounding_box(img)
        if bbox:
            img = img.crop(bbox)
        
        # 2. Add padding to make it square
        width, height = img.size
        size = max(width, height) + 20  # Add padding
        new_img = Image.new("L", (size, size), 0)
        paste_x = (size - width) // 2
        paste_y = (size - height) // 2
        new_img.paste(img, (paste_x, paste_y))
        img = new_img
        
        # 3. Resize to MNIST format (28x28)
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # 4. Increase contrast to match MNIST more closely
        img = T.functional.adjust_contrast(img, 2.0)
        
        # 5. Normalize pixel values (MNIST is normalized to [0,1])
        img_array = np.array(img)
        if img_array.max() > 0:  # Avoid division by zero
            img_array = img_array / 255.0
        
        # Save processed image for debugging (optional)
        # Can be removed in production
        processed_image = Image.fromarray((img_array * 255).astype(np.uint8))
        processed_image.save("processed_digit.png")
        
        # Convert to tensor for model input
        tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = model(tensor)
            probabilities = F.softmax(output, dim=1)
            prediction = output.argmax().item()
            confidence = probabilities[0][prediction].item() * 100
        
        # Update UI
        self.result_label.config(text=f"Predicted: {prediction} ({confidence:.1f}% confidence)")
        self.status_bar.config(text=f"Prediction: {prediction} with {confidence:.1f}% confidence")
        
        # Send to Arduino if connected
        if arduino:
            try:
                arduino.write(f"{prediction}\n".encode())
                self.status_bar.config(text=f"Prediction: {prediction} - Sent to Arduino")
            except serial.SerialException as e:
                self.status_bar.config(text=f"Arduino error: {e}")
                
    def get_bounding_box(self, img):
        """Find bounding box of the digit in the image"""
        img_array = np.array(img)
        rows = np.any(img_array, axis=1)
        cols = np.any(img_array, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None  # No content found
            
        # Find the boundaries of the digit
        top = np.argmax(rows)
        bottom = len(rows) - np.argmax(rows[::-1]) - 1
        left = np.argmax(cols)
        right = len(cols) - np.argmax(cols[::-1]) - 1
        
        # Add a small margin
        margin = 5
        return (max(0, left - margin),
                max(0, top - margin),
                min(img.width, right + margin),
                min(img.height, bottom + margin))

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitDrawingApp(root)
    root.mainloop()
    
    # Close Arduino connection when app closes
    if arduino:
        arduino.close()
