# Handwritten Digit Recognition with Arduino Display

A Python-based application that recognizes handwritten digits drawn on a digital canvas and displays them on a seven-segment display connected to an Arduino Nano. The project combines computer vision, machine learning, and hardware integration.

## Features

- Interactive drawing canvas for inputting handwritten digits
- CNN model trained on MNIST dataset for digit recognition
- Real-time preprocessing pipeline to improve recognition accuracy
- Live preview of the processed 28x28 MNIST-format image
- Confidence scores for predictions
- Direct output to a seven-segment display via Arduino Nano

## Technical Details

The system uses a PyTorch CNN model trained on the MNIST dataset to classify handwritten digits. The application includes:

- Custom Tkinter-based drawing interface
- Advanced image preprocessing pipeline that crops, centers, and normalizes input
- Serial communication with Arduino Nano
- Seven-segment display control for physical output

## Requirements

- Python 3.x with PyTorch, Pillow, NumPy, and pyserial
- Arduino Nano with appropriate connections to a seven-segment display
- Trained model file (`digit_cnn.pth`)

## Setup

1. Clone the repository
2. Connect your Arduino Nano to a common-cathode (or common-anode) seven-segment display using pins 2-8
3. Upload the Arduino code to your Nano
4. Run the training script to generate the model file (or use the pre-trained model)
5. Update the serial port in `main.py` to match your Arduino connection
6. Run `main.py` to start the application

## Usage

1. Draw a digit (0-9) on the black canvas
2. Click "Predict" to process and recognize the digit
3. The predicted digit will appear on the seven-segment display connected to your Arduino
4. The application will show preprocessing details and confidence scores

## Project Structure

- `main.py`: The main application with Tkinter interface and prediction logic
- `trainig_model_(jupyter_notebook).ipynb`: CNN model architecture and training code
- `arduino_nano_code.ino`: Arduino sketch for controlling the seven-segment display
- `wiring.txt`: The wiring for the seven-segment display


