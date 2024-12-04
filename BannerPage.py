import tkinter as tk
from PIL import Image, ImageTk
from userInterface import FarmManagementSystem
import cv2

# Create the main window
root = tk.Tk()
root.title("FM System by OD")
root.state('zoomed')  # Make the window full screen
root.minsize(640, 480)  # Set the minimum size of the window
root.resizable(True, True)  # Enable the maximize and minimize buttons

# Load the logo image
logo = Image.open("Image/logo.png")
logo = logo.resize((32, 32))  # Resize the logo to 32x32 pixels
logo = ImageTk.PhotoImage(logo)
root.iconphoto(False, logo)

# Load the logo title image
logo_title = Image.open("Image/logo_title.png")
logo_title = logo_title.resize((400, 200))  # Resize the logo title to 400x200 pixels
logo_title = ImageTk.PhotoImage(logo_title)

# Create a frame to hold the image and button
frame = tk.Frame(root, bg="white")
frame.pack(fill=tk.BOTH, expand=True)

# Load the image
image = Image.open("Image/image.jpg")
image = image.resize((root.winfo_screenwidth(), root.winfo_screenheight()))
image = ImageTk.PhotoImage(image)

# Create a label to display the image
image_label = tk.Label(frame, image=image)
image_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create a label to display the logo title
logo_title_label = tk.Label(image_label, image=logo_title, bg="white")
logo_title_label.image = logo_title
logo_title_label.place(relx=0.5, rely=0.15, anchor=tk.CENTER)

# Create a button to start the application
def start_application():
    root.destroy()
    # Create a new window for the application
    fms= FarmManagementSystem()
    fms.create_gui()

start_button = tk.Button(image_label, text="START", command=start_application, font=("Arial", 24, "bold"), bg="#800080", fg="white", width=10, height=2)
start_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Start the Tkinter event loop
root.mainloop()