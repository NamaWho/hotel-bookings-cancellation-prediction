import tkinter as tk
from tkinter import messagebox
import numpy as np
import pickle

# Load the model from the pickle file
with open('booking_cancellations_classifier.pkl', 'rb') as model_file:
    pipeline = pickle.load(model_file)

def predict():
    try:
        # Retrieve user inputs from the entry boxes
        lead_time = float(entry_lead_time.get())
        arrival_year = int(entry_arrival_year.get())
        arrival_month = int(entry_arrival_month.get())
        stays_weekend_nights = int(entry_stays_weekend_nights.get())
        stays_week_nights = int(entry_stays_week_nights.get())
        adults = int(entry_adults.get())
        children = float(entry_children.get())
        babies = int(entry_babies.get())
        previous_cancellations = int(entry_previous_cancellations.get())
        is_repeated_guest = int(entry_is_repeated_guest.get())

        # Create the feature array for the model input
        features = np.array([[lead_time, arrival_year, arrival_month, stays_weekend_nights,
                              stays_week_nights, adults, children, babies, previous_cancellations,
                              is_repeated_guest]])
        
        # Perform prediction using the pipeline
        prediction = pipeline.predict(features)[0]
        
        # Show the prediction result in a message box
        result = "Cancelled" if prediction == 1 else "Not Cancelled"
        messagebox.showinfo("Prediction", f"The predicted class is: {result}")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numbers for all fields.")

# Initialize the Tkinter window
root = tk.Tk()
root.title("Hotel Booking Cancellations Classifier")

# Create a welcome message and explain the purpose of the app
welcome_message = tk.Label(root, text="Welcome to the Hotel Booking Cancellations Classifier!")
welcome_message.grid(row=0, column=0, columnspan=2)

explanation_message = tk.Label(root, text="Please enter the following information to predict whether a booking will be cancelled:")
explanation_message.grid(row=1, column=0, columnspan=2)

# Create input fields for user to enter the features
label_lead_time = tk.Label(root, text="Lead Time:")
label_lead_time.grid(row=2, column=0)
entry_lead_time = tk.Entry(root)
entry_lead_time.grid(row=2, column=1)

label_arrival_year = tk.Label(root, text="Arrival Year:")
label_arrival_year.grid(row=3, column=0)
entry_arrival_year = tk.Entry(root)
entry_arrival_year.grid(row=3, column=1)

label_arrival_month = tk.Label(root, text="Arrival Month (1-12):")
label_arrival_month.grid(row=4, column=0)
entry_arrival_month = tk.Entry(root)
entry_arrival_month.grid(row=4, column=1)

label_stays_weekend_nights = tk.Label(root, text="Stays in Weekend Nights:")
label_stays_weekend_nights.grid(row=5, column=0)
entry_stays_weekend_nights = tk.Entry(root)
entry_stays_weekend_nights.grid(row=5, column=1)

label_stays_week_nights = tk.Label(root, text="Stays in Week Nights:")
label_stays_week_nights.grid(row=6, column=0)
entry_stays_week_nights = tk.Entry(root)
entry_stays_week_nights.grid(row=6, column=1)

label_adults = tk.Label(root, text="Number of Adults:")
label_adults.grid(row=7, column=0)
entry_adults = tk.Entry(root)
entry_adults.grid(row=7, column=1)

label_children = tk.Label(root, text="Number of Children:")
label_children.grid(row=8, column=0)
entry_children = tk.Entry(root)
entry_children.grid(row=8, column=1)

label_babies = tk.Label(root, text="Number of Babies:")
label_babies.grid(row=9, column=0)
entry_babies = tk.Entry(root)
entry_babies.grid(row=9, column=1)

label_previous_cancellations = tk.Label(root, text="Previous Cancellations:")
label_previous_cancellations.grid(row=10, column=0)
entry_previous_cancellations = tk.Entry(root)
entry_previous_cancellations.grid(row=10, column=1)

label_is_repeated_guest = tk.Label(root, text="Is Repeated Guest (1=Yes, 0=No):")
label_is_repeated_guest.grid(row=11, column=0)
entry_is_repeated_guest = tk.Entry(root)
entry_is_repeated_guest.grid(row=11, column=1)

# Create a predict button
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.grid(row=10, column=0, columnspan=2)

# Run the Tkinter event loop
root.mainloop()

