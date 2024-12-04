import cv2
import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import cvzone
import math
import pygame
import time
from ClassNamesofObject import CN, ACN
from disease import PlantDiseasePredictor
import numpy as np
import os

def ringSound():
    pygame.mixer.init()
    pygame.mixer.music.load("Audio/alarm.mp3")  # Replace with your own sound file
    pygame.mixer.music.play()
    time.sleep(1)  # Play the sound for 1 second
    pygame.mixer.music.stop()

def capture_leaf_image():
    # Create a history folder if it doesn't exist
    history_folder = 'history'
    if not os.path.exists(history_folder):
        os.makedirs(history_folder)

    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)
    image_path =  None
    while True:
        print("working...")
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Display the captured frame
        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2_im)
        imgtk = ImageTk.PhotoImage(image=img)
        label.config(image=imgtk)
        label.image = imgtk

        # Check for leaf detection
        if detect_leaf(frame):
            # Save the image in the history folder
            image_path = os.path.join(history_folder, 'leaf_snapshot.png')
            cv2.imwrite(image_path, frame)
            print(f"Leaf detected! Image saved at: {image_path}")

            break

    cap.release()
    cv2.destroyAllWindows()

    return image_path

def detect_leaf(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for green color in HSV
    lower_green = np.array([35, 100, 100])  # Lower bound for green
    upper_green = np.array([85, 255, 255])  # Upper bound for green

    # Create a mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours are found
    if contours:
        # Optionally, you can filter contours based on area
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold to consider as a leaf
                return True  # Leaf detected

    return False

class FarmManagementSystem:
    def __init__(self):
        self.checkSelect = 0
        self.checkAlert = 0
        self.selected_className = []
        self.cap = None
        self.model = None
        self.classNames = CN
        self.checkClassNameforPrint = ACN
        self.disease_prediction_active = False

    def start_webcam(self):
        print("Starting webcam...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera")
        else:
            print("Camera is opened")
        if self.cap is None:
            print("Error: Unable to open webcam")
            return
        print("Webcam started successfully")
        if self.disease_prediction_active == False:
            self.model = YOLO("Yolo-Weights/yolov8n.pt")
            self.update_label()
        else:
            self.model = None

    def alert_webcam(self):
        self.checkAlert = 1
        self.stop_webcam()
        self.start_webcam()

    def update_alert_label(self):
        if self.cap is None:
            print("Error: Webcam not started")
            return
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Unable to read frame from webcam")
            return
        frame = cv2.resize(frame, (640, 480))
        results = self.model(frame, stream=True)
        detected_objects = {}
        for r in results:
            if self.checkSelect == 1:
                self.checkClassNameforPrint = self.selected_className
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cls = int(box.cls[0])
                class_name = self.classNames[cls]

                if class_name in detected_objects:
                    detected_objects[class_name] += 1
                else:
                    detected_objects[class_name] = 1

                if class_name in self.checkClassNameforPrint:  # Add this condition
                    cvzone.cornerRect(frame, (x1, y1, w, h))
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cvzone.putTextRect(frame, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2_im)
        imgtk = ImageTk.PhotoImage(image=img)
        label.config(image=imgtk)
        label.image = imgtk

        display_text = ""
        for class_name, count in detected_objects.items():
            if class_name in self.checkClassNameforPrint:  # Add this condition
                display_text += f"{class_name}: {count}\n"
                print(f"{class_name}: {count}")  # Add this line to print in command panel
                ringSound()  # Call the ringSound function to play a sound

        display_label.delete(1.0, tk.END)  # Clear the widget
        display_label.insert(tk.END, display_text)  # Insert the new text

        if self.cap is not None:
            root.after(1, self.update_alert_label)  # Update the label 1000 times per second

    def update_label(self):
        if self.cap is None:
            print("Error: Webcam not started")
            return
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Unable to read frame from webcam")
            return
        frame = cv2.resize(frame, (640, 480))
        results = self.model(frame, stream=True)
        detected_objects = {}
        for r in results:
            if self.checkSelect == 1:
                self.checkClassNameforPrint = self.selected_className
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cls = int(box.cls[0])
                class_name = self.classNames[cls]

                if class_name in detected_objects:
                    detected_objects[class_name] += 1
                else:
                    detected_objects[class_name] = 1

                if class_name in self.checkClassNameforPrint:  # Add this condition
                    cvzone.cornerRect(frame, (x1, y1, w, h))
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cvzone.putTextRect(frame, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2_im)
        imgtk = ImageTk.PhotoImage(image=img)
        label.config(image=imgtk)
        label.image = imgtk

        display_text = ""
        for class_name, count in detected_objects.items():
            if class_name in self.checkClassNameforPrint:  # Add this condition
                display_text += f"{class_name}: {count}\n"
                print(f"{class_name}: {count}")
                if self.checkAlert == 1:
                    print("playing sound...")
                    ringSound()  # Add this line to print in command panel
        display_label.delete(1.0, tk.END)  # Clear the widget
        display_label.insert(tk.END, display_text)  # Insert the new text

        if self.cap is not None:
            root.after(1, self.update_label)  # Update the label 100 times per second

    def stop_webcam(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            label.config(image=None)
            label.config(bg="black")

    def unalert_webcam(self):
        if self.checkAlert == 1:
            self.checkAlert = 0
            self.stop_webcam()
            self.start_webcam()

    def start_disease_prediction(self):
        self.stop_webcam()
        self.disease_prediction_active = True
        print("Disease prediction started.")
        result = capture_leaf_image()
        predictor = PlantDiseasePredictor()
        display_text = predictor.predict(result)
        display_label.delete(1.0, tk.END)
        display_label.insert(tk.END, display_text)


    def stop_disease_prediction(self):
        self.stop_webcam()
        self.disease_prediction_active = False
        print("Disease prediction stopped.")
        self.start_webcam()

    def create_gui(self):
        global root, label, display_label
        root = tk.Tk()
        root.title("FM System by OD")
        root.state('zoomed')  # Make the window full screen
        root.minsize(640, 480)  # Set the minimum size of the window
        root.resizable(True, True)  # Enable the maximize and minimize buttons

        logo = Image.open("Image/logo.png")
        logo = logo.resize((32, 32))  # Resize the logo to 32x32 pixels
        logo = ImageTk.PhotoImage(logo)
        root.iconphoto(False, logo)

        main_frame = tk.Frame(root, bg="white")
        main_frame.pack(fill=tk.BOTH, expand=True)

        column1_frame = tk.Frame(main_frame, width=750, height=860, bg="white")
        column1_frame.pack(side=tk.LEFT, padx=10, pady=10)

        column1_label = tk.Label(column1_frame, text="Screening Portion", font=("Arial", 24))
        column1_label.pack(pady=10)

        webcam_frame = tk.Frame(column1_frame, width=640, height=480, bg="black", highlightbackground="black",
                                highlightthickness=2)
        webcam_frame.pack(padx=10, pady=10)

        label = tk.Label(webcam_frame, bg="black")
        label.place(x=0, y=0, width=640, height=480)

        button_frame = tk.Frame(column1_frame, width=640, height=100)
        button_frame.pack(padx=10, pady=10)

        # First row of buttons
        first_row = tk.Frame(button_frame)
        first_row.pack(side=tk.TOP)

        start_button = tk.Button(first_row, text="Start Webcam", command=self.start_webcam)
        start_button.pack(side=tk.LEFT, padx=10)

        stop_button = tk.Button(first_row, text="Stop Webcam", command=self.stop_webcam)
        stop_button.pack(side=tk.LEFT, padx=10)

        alert_button = tk.Button(first_row, text="Alert Webcam", command=self.alert_webcam)
        alert_button.pack(side=tk.LEFT, padx=10)

        unalert_button = tk.Button(first_row, text="UnAlert Webcam", command=self.unalert_webcam)
        unalert_button.pack(side=tk.LEFT, padx=10)

        # Second row of buttons
        second_row = tk.Frame(button_frame)
        second_row.pack(side=tk.TOP, pady=(10, 0))  # Add space above the second row

        disease_start_button = tk.Button(second_row, text="Start Disease Prediction",
                                         command=self.start_disease_prediction)
        disease_start_button.pack(side=tk.LEFT, padx=10)

        disease_stop_button = tk.Button(second_row, text="Stop Disease Prediction",
                                        command=self.stop_disease_prediction)
        disease_stop_button.pack(side=tk.LEFT, padx=10)

        column2_frame = tk.Frame(main_frame, width=750, height=860, bg="white")
        column2_frame.pack(side=tk.LEFT, padx=10, pady=10)

        column2_label = tk.Label(column2_frame, text="Output Portion", font=("Arial", 24))
        column2_label.pack(pady=10)

        output_frame = tk.Frame(column2_frame, width=750, height=840)
        output_frame.pack(padx=10, pady=10)

        display_area_frame = tk.Frame(output_frame, width=750, height=300, bg="white")
        display_area_frame.pack(side=tk.TOP, padx=10, pady=10)

        display_area_label = tk.Label(display_area_frame, text="Display Area", font=("Arial", 18))
        display_area_label.pack(pady=10)

        display_text_frame = tk.Frame(display_area_frame, width=700, height=200, bg="white")
        display_text_frame.pack(padx=10, pady=10)

        scrollbar = tk.Scrollbar(display_text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        display_label = tk.Text(display_text_frame, width=80, height=10, font=("Arial", 14),
                                yscrollcommand=scrollbar.set)
        display_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=display_label.yview)

        instruction_area_frame = tk.Frame(output_frame, width=750, height=300, bg="white")
        instruction_area_frame.pack(side=tk.TOP, padx=10, pady=10)

        instruction_area_label = tk.Label(instruction_area_frame, text="Instruction Area", font=("Arial", 18))
        instruction_area_label.grid(row=0, column=0, columnspan=6, padx=10, pady=10)

        scrollbar_frame = tk.Frame(instruction_area_frame)
        scrollbar_frame.grid(row=1, column=0, columnspan=6, padx=10, pady=10)

        scrollbar = tk.Scrollbar(scrollbar_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        canvas = tk.Canvas(scrollbar_frame, width=700, height=200, yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=canvas.yview)

        frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=frame, anchor='nw')
        frame.pack(fill=tk.BOTH, expand=True)

        # Update the scrollregion attribute of the Canvas widget
        frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

        # Add this line to update the scrollregion attribute again
        canvas.update_idletasks()

        # Set the scrollregion attribute to a specific value
        canvas.config(scrollregion=(0, 0, 0, 1000))

        self.classNames2 = ACN

        self.selected_className = []
        var_dict = {}

        for i, className in enumerate(self.classNames2):
            var = tk.IntVar()
            var_dict[className] = var
            checkbox = tk.Checkbutton(frame, text=className, variable=var)
            checkbox.grid(row=i // 6, column=i % 6, padx=10, pady=10)

        def update_selected_className():
            self.selected_className = [className for className, var in var_dict.items() if var.get()]
            if self.selected_className == []:
                messagebox.showinfo("Select Something Please...", "Select Something Please...")
            else:
                self.checkSelect = 1
                print("Selected classes updated:", self.selected_className)
                messagebox.showinfo("Selected", "Selected")

        def reset_selected_className():
            self.selected_className = ACN
            print("Selected classes updated:", self.selected_className)

        update_button = tk.Button(instruction_area_frame, text="Update", command=update_selected_className)
        update_button.grid(row=2, column=0, columnspan=6, padx=10, pady=10)
        reset_button = tk.Button(instruction_area_frame, text="Reset", command=reset_selected_className)
        reset_button.grid(row=2, column=1, columnspan=6, padx=10, pady=10)

        frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

        root.mainloop()

if __name__ == "__main__":
    fms = FarmManagementSystem()
    fms.create_gui()