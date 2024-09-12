import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import pickle
from ttkbootstrap import Style

# Load the model from the pickle file
with open('booking_cancellations_classifier.pkl', 'rb') as model_file:
    pipeline = pickle.load(model_file)

def predict():
    try:
        # Retrieve user inputs from the entry boxes
        lead_time = float(entry_lead_time.get())
        arrival_year = int(entry_arrival_year.get())
        arrival_month = int(entry_arrival_month.get())
        arrival_day_of_month = int(entry_arrival_day_of_month.get())
        stays_weekend_nights = int(entry_stays_weekend_nights.get())
        stays_week_nights = int(entry_stays_week_nights.get())
        adults = int(entry_adults.get())
        children = float(entry_children.get())
        previous_cancellations = int(entry_previous_cancellations.get())
        is_repeated_guest = int(entry_is_repeated_guest.get())
        adr = float(entry_adr.get())
        previous_bookings_not_canceled = int(entry_previous_bookings_not_canceled.get())
        total_of_special_requests = int(entry_total_of_special_requests.get())
        days_in_waiting_list = int(entry_days_in_waiting_list.get())
        booking_changes = int(entry_booking_changes.get())

        # Retrieve categorical inputs from the dropdowns
        meal = meal_var.get()
        market_segment = market_segment_var.get()
        distribution_channel = distribution_channel_var.get()
        reserved_room_type = reserved_room_type_var.get()
        deposit_type = deposit_type_var.get()
        customer_type = customer_type_var.get()
        agent = int(entry_agent.get())
        company = int(entry_company.get())
        
        # Create a dictionary of features
        features_dict = {
            'LeadTime': [lead_time],
            'ArrivalDateYear': [arrival_year],
            'ArrivalDateMonth': [arrival_month],
            'ArrivalDateDayOfMonth': [arrival_day_of_month],
            'StaysInWeekendNights': [stays_weekend_nights],
            'StaysInWeekNights': [stays_week_nights],
            'Adults': [adults],
            'Children': [children],
            'ADR': [adr],
            'PreviousCancellations': [previous_cancellations],
            'IsRepeatedGuest': [is_repeated_guest],
            'Meal': [meal],
            'MarketSegment': [market_segment],
            'DistributionChannel': [distribution_channel],
            'ReservedRoomType': [reserved_room_type],
            'DepositType': [deposit_type],
            'CustomerType': [customer_type],
            'Agent': [agent],
            'Company': [company],
            'ReservationStatus': [''],
            'ReservationStatusDate': [''],
            'Babies': [0], 
            'Hotel': [''],
            'PreviousBookingsNotCanceled': [previous_bookings_not_canceled],
            'TotalOfSpecialRequests': [total_of_special_requests],
            'DaysInWaitingList': [days_in_waiting_list],
            'BookingChanges': [booking_changes]
        }
        
        # Convert the features to a DataFrame
        features_df = pd.DataFrame(features_dict)

        # Perform prediction using the model
        prediction = pipeline.predict(features_df)[0]
        
        # Show the prediction result in a message box
        result = "Cancelled" if prediction == 1 else "Not Cancelled"
        messagebox.showinfo("Prediction", f"The predicted class is: {result}")
    except ValueError as ve:
        print(f"ValueError: {ve}")
        messagebox.showerror("Error", f"ValueError: {ve}")
    except Exception as e:
        print(f"Exception: {e}")
        messagebox.showerror("Error", f"An error occurred: {e}")

# Initialize the Tkinter window with ttkbootstrap style
root = tk.Tk()
style = Style(theme='flatly')
root.title("🏨 Hotel Booking Cancellations Classifier 🏨")
root.geometry("800x600")

# Create a main frame
main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=1)

# Create a canvas
canvas = tk.Canvas(main_frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

# Add a scrollbar to the canvas
scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Configure the canvas
canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# Create another frame inside the canvas
content_frame = ttk.Frame(canvas)

# Add that frame to a window in the canvas
canvas.create_window((400, 0), window=content_frame, anchor="n")

# Make the content frame expandable
content_frame.columnconfigure(0, weight=1)

# Bind mousewheel to scroll
def _on_mousewheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")

canvas.bind_all("<MouseWheel>", _on_mousewheel)

# Create a welcome message and explain the purpose of the app
welcome_message = ttk.Label(content_frame, text="Welcome to the Hotel Booking Cancellations Classifier!", font=("Helvetica", 16, "bold"))
welcome_message.grid(row=0, column=0, pady=(20, 10))

explanation_message = ttk.Label(content_frame, text="Please enter the following information to predict whether a booking will be cancelled:", wraplength=700)
explanation_message.grid(row=1, column=0, pady=(0, 20))

# Create a function to add input fields with consistent styling and default values
def add_input_field(parent, row, text, default=""):
    frame = ttk.Frame(parent)
    frame.grid(row=row, column=0, sticky="ew", pady=5)
    frame.columnconfigure(1, weight=1)
    
    label = ttk.Label(frame, text=text, width=25, anchor="e")
    label.grid(row=0, column=0, padx=(0, 10))
    
    entry = ttk.Entry(frame)
    entry.grid(row=0, column=1, sticky="ew")
    entry.insert(0, default)
    
    return entry

# Create input fields for user to enter the features
entry_lead_time = add_input_field(content_frame, 2, "Lead Time:", "0")
entry_arrival_year = add_input_field(content_frame, 3, "Arrival Year:", "2017")
entry_arrival_month = add_input_field(content_frame, 4, "Arrival Month (1-12):", "1")
entry_arrival_day_of_month = add_input_field(content_frame, 5, "Arrival Day of Month:", "1")
entry_stays_weekend_nights = add_input_field(content_frame, 6, "Stays in Weekend Nights:", "1")
entry_stays_week_nights = add_input_field(content_frame, 7, "Stays in Week Nights:", "2")
entry_adults = add_input_field(content_frame, 8, "Number of Adults:", "1")
entry_children = add_input_field(content_frame, 9, "Number of Children:", "0")
entry_previous_cancellations = add_input_field(content_frame, 10, "Previous Cancellations:", "0")
entry_is_repeated_guest = add_input_field(content_frame, 11, "Is Repeated Guest (1=Yes, 0=No):", "0")
entry_adr = add_input_field(content_frame, 12, "Average Daily Rate:", "100")

# Categorical feature dropdowns
def add_dropdown(parent, row, text, variable, options):
    frame = ttk.Frame(parent)
    frame.grid(row=row, column=0, sticky="ew", pady=5)
    frame.columnconfigure(1, weight=1)
    
    label = ttk.Label(frame, text=text, width=25, anchor="e")
    label.grid(row=0, column=0, padx=(0, 10))
    
    dropdown = ttk.Combobox(frame, textvariable=variable, values=options, state="readonly")
    dropdown.grid(row=0, column=1, sticky="ew")
    dropdown.set(options[0])
    
    return dropdown

meal_var = tk.StringVar()
add_dropdown(content_frame, 13, "Meal:", meal_var, ["BB", "FB", "HB", "SC", "Undefined"])

market_segment_var = tk.StringVar()
add_dropdown(content_frame, 14, "Market Segment:", market_segment_var, ["Direct", "Corporate", "Online TA", "Offline TA/TO", "Complementary", "Groups", "Undefined"])

distribution_channel_var = tk.StringVar()
add_dropdown(content_frame, 15, "Distribution Channel:", distribution_channel_var, ["TA/TO", "Corporate", "Direct", "GDS", "Undefined"])

reserved_room_type_var = tk.StringVar()
add_dropdown(content_frame, 16, "Reserved Room Type:", reserved_room_type_var, ["A", "B", "C", "D", "E"])

deposit_type_var = tk.StringVar()
add_dropdown(content_frame, 17, "Deposit Type:", deposit_type_var, ["No Deposit", "Non Refund", "Refundable"])

customer_type_var = tk.StringVar()
add_dropdown(content_frame, 18, "Customer Type:", customer_type_var, ["Transient", "Contract", "Transient-Party", "Group"])

entry_agent = add_input_field(content_frame, 19, "Agent:", "9")
entry_company = add_input_field(content_frame, 20, "Company:", "40")
entry_previous_bookings_not_canceled = add_input_field(content_frame, 21, "Previous Bookings Not Canceled:", "0")
entry_total_of_special_requests = add_input_field(content_frame, 22, "Total of Special Requests:", "0")
entry_days_in_waiting_list = add_input_field(content_frame, 23, "Days in Waiting List:", "0")
entry_booking_changes = add_input_field(content_frame, 24, "Booking Changes:", "0")

# Create a predict button with custom styling
predict_button = ttk.Button(content_frame, text="Predict", command=predict, style="Accent.TButton")
predict_button.grid(row=25, column=0, pady=20)

# Run the Tkinter event loop
root.mainloop()