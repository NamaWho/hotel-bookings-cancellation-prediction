import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib  # To load your pre-trained pipeline model

# Load the trained pipeline model (replace with your actual file path)
# pipeline = joblib.load('path_to_your_pipeline_model.pkl')

def predict():
    # Retrieve user inputs from the entry boxes
    try:
        # Add the number of features you need from your model (just using two as an example)
        feature_1 = float(entry_feature_1.get())
        feature_2 = float(entry_feature_2.get())
        # Add more features as needed, following the structure of your pipeline

        # Create the feature array for the model input
        features = np.array([[feature_1, feature_2]])  # Adjust this for your model's expected input

        # Perform prediction using the pipeline
        # prediction = pipeline.predict(features)[0]
        
        # Show the prediction result in a message box
        messagebox.showinfo("Prediction", f"The predicted class is: {2}")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numbers for all fields.")

# Initialize the Tkinter window
root = tk.Tk()
root.title("ML Model Prediction Interface")

# Create input fields for features
label_feature_1 = tk.Label(root, text="Feature 1:")
label_feature_1.grid(row=0, column=0)
entry_feature_1 = tk.Entry(root)
entry_feature_1.grid(row=0, column=1)

label_feature_2 = tk.Label(root, text="Feature 2:")
label_feature_2.grid(row=1, column=0)
entry_feature_2 = tk.Entry(root)
entry_feature_2.grid(row=1, column=1)

# Add more input fields as necessary, depending on your model's feature set
# label_feature_n = tk.Label(root, text="Feature N:")
# label_feature_n.grid(row=n, column=0)
# entry_feature_n = tk.Entry(root)
# entry_feature_n.grid(row=n, column=1)

# Create a predict button
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.grid(row=10, column=0, columnspan=2)

# Run the Tkinter event loop
root.mainloop()
