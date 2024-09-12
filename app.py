import tkinter as tk
from tkinter import messagebox
import pandas as pd
import pickle
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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
        deposit_type = deposit_type_var.get()  # Assuming there's a variable for 'DepositType'
        customer_type = customer_type_var.get()  # Assuming there's a variable for 'CustomerType'
        agent = int(entry_agent.get())  # Assuming 'Agent' is provided as input
        company = int(entry_company.get())  # Assuming 'Company' is provided as input
        
        # Create a dictionary of features, corresponding to column names expected by the model
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
        
        # Convert the features to a DataFrame (this matches the expected format for the model)
        features_df = pd.DataFrame(features_dict)

        # Perform prediction using the model
        prediction = pipeline.predict(features_df)[0]
        
        # Show the prediction result in a message box
        result = "Cancelled" if prediction == 1 else "Not Cancelled"
        messagebox.showinfo("Prediction", f"The predicted class is: {result}")
    except ValueError as ve:
        # Catch ValueError specifically and print detailed information
        print(f"ValueError: {ve}")
        messagebox.showerror("Error", f"ValueError: {ve}")
    except Exception as e:
        # Catch any other exceptions and show detailed error message
        print(f"Exception: {e}")
        messagebox.showerror("Error", f"An error occurred: {e}")


# Initialize the Tkinter window
root = tk.Tk()
root.title("🏨 Hotel Booking Cancellations Classifier 🏨")

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
entry_lead_time.insert(0, "0")  # Default value

label_arrival_year = tk.Label(root, text="Arrival Year:")
label_arrival_year.grid(row=3, column=0)
entry_arrival_year = tk.Entry(root)
entry_arrival_year.grid(row=3, column=1)
entry_arrival_year.insert(0, "2017")  # Default value

label_arrival_month = tk.Label(root, text="Arrival Month (1-12):")
label_arrival_month.grid(row=4, column=0)
entry_arrival_month = tk.Entry(root)
entry_arrival_month.grid(row=4, column=1)
entry_arrival_month.insert(0, "1")  # Default value

label_arrival_day_of_month = tk.Label(root, text="Arrival Day of Month:")  # Add Day of Month label and entry
label_arrival_day_of_month.grid(row=5, column=0)
entry_arrival_day_of_month = tk.Entry(root)
entry_arrival_day_of_month.grid(row=5, column=1)
entry_arrival_day_of_month.insert(0, "1")  # Default value

label_stays_weekend_nights = tk.Label(root, text="Stays in Weekend Nights:")
label_stays_weekend_nights.grid(row=6, column=0)
entry_stays_weekend_nights = tk.Entry(root)
entry_stays_weekend_nights.grid(row=6, column=1)
entry_stays_weekend_nights.insert(0, "1")  # Default value

label_stays_week_nights = tk.Label(root, text="Stays in Week Nights:")
label_stays_week_nights.grid(row=7, column=0)
entry_stays_week_nights = tk.Entry(root)
entry_stays_week_nights.grid(row=7, column=1)
entry_stays_week_nights.insert(0, "2")  # Default value

label_adults = tk.Label(root, text="Number of Adults:")
label_adults.grid(row=8, column=0)
entry_adults = tk.Entry(root)
entry_adults.grid(row=8, column=1)
entry_adults.insert(0, "1")  # Default value

label_children = tk.Label(root, text="Number of Children:")
label_children.grid(row=9, column=0)
entry_children = tk.Entry(root)
entry_children.grid(row=9, column=1)
entry_children.insert(0, "0")  # Default value

label_previous_cancellations = tk.Label(root, text="Previous Cancellations:")
label_previous_cancellations.grid(row=11, column=0)
entry_previous_cancellations = tk.Entry(root)
entry_previous_cancellations.grid(row=11, column=1)
entry_previous_cancellations.insert(0, "0")  # Default value

label_is_repeated_guest = tk.Label(root, text="Is Repeated Guest (1=Yes, 0=No):")
label_is_repeated_guest.grid(row=12, column=0)
entry_is_repeated_guest = tk.Entry(root)
entry_is_repeated_guest.grid(row=12, column=1)
entry_is_repeated_guest.insert(0, "0")  # Default value

label_adr = tk.Label(root, text="Average Daily Rate:")
label_adr.grid(row=13, column=0)
entry_adr = tk.Entry(root)
entry_adr.grid(row=13, column=1)
entry_adr.insert(0, "100")  # Default value

# Categorical feature dropdowns
label_meal = tk.Label(root, text="Meal:")
label_meal.grid(row=14, column=0)
meal_var = tk.StringVar(root)
meal_var.set("BB")  # Default value
meal_options = ["BB", "FB", "HB", "SC", "Undefined"]  # Add actual options
meal_menu = tk.OptionMenu(root, meal_var, *meal_options)
meal_menu.grid(row=14, column=1)

label_market_segment = tk.Label(root, text="Market Segment:")
label_market_segment.grid(row=15, column=0)
market_segment_var = tk.StringVar(root)
market_segment_var.set("Online TA")  # Default value
market_segment_options = ["Direct", "Corporate", "Online TA", "Offline TA/TO", "Complementary", "Groups", "Undefined"]
market_segment_menu = tk.OptionMenu(root, market_segment_var, *market_segment_options)
market_segment_menu.grid(row=15, column=1)

label_distribution_channel = tk.Label(root, text="Distribution Channel:")
label_distribution_channel.grid(row=16, column=0)
distribution_channel_var = tk.StringVar(root)
distribution_channel_var.set("TA/TO")  # Default value
distribution_channel_options = ["TA/TO", "Corporate", "Direct", "GDS", "Undefined"]
distribution_channel_menu = tk.OptionMenu(root, distribution_channel_var, *distribution_channel_options)
distribution_channel_menu.grid(row=16, column=1)

label_reserved_room_type = tk.Label(root, text="Reserved Room Type:")
label_reserved_room_type.grid(row=17, column=0)
reserved_room_type_var = tk.StringVar(root)
reserved_room_type_var.set("A")  # Default value
reserved_room_type_options = ["A", "B", "C", "D", "E"]  # Actual room types
reserved_room_type_menu = tk.OptionMenu(root, reserved_room_type_var, *reserved_room_type_options)
reserved_room_type_menu.grid(row=17, column=1)

label_deposit_type = tk.Label(root, text="Deposit Type:")
label_deposit_type.grid(row=18, column=0)
deposit_type_var = tk.StringVar(root)
deposit_type_var.set("No Deposit")  # Default value
deposit_type_options = ["No Deposit", "Non Refund", "Refundable"]
deposit_type_menu = tk.OptionMenu(root, deposit_type_var, *deposit_type_options)
deposit_type_menu.grid(row=18, column=1)

label_customer_type = tk.Label(root, text="Customer Type:")
label_customer_type.grid(row=19, column=0)
customer_type_var = tk.StringVar(root)
customer_type_var.set("Transient")  # Default value
customer_type_options = ["Transient", "Contract", "Transient-Party", "Group"]
customer_type_menu = tk.OptionMenu(root, customer_type_var, *customer_type_options)
customer_type_menu.grid(row=19, column=1)

label_agent = tk.Label(root, text="Agent:")
label_agent.grid(row=20, column=0)
entry_agent = tk.Entry(root)
entry_agent.grid(row=20, column=1)
entry_agent.insert(0, "9")  # Default value

label_company = tk.Label(root, text="Company:")
label_company.grid(row=21, column=0)
entry_company = tk.Entry(root)
entry_company.grid(row=21, column=1)
entry_company.insert(0, "40")  # Default value

# add {'PreviousBookingsNotCanceled', 'TotalOfSpecialRequests', 'DaysInWaitingList', 'BookingChanges'}

previous_bookings_not_canceled = tk.Label(root, text="Previous Bookings Not Canceled:")
previous_bookings_not_canceled.grid(row=22, column=0)
entry_previous_bookings_not_canceled = tk.Entry(root)
entry_previous_bookings_not_canceled.grid(row=22, column=1)
entry_previous_bookings_not_canceled.insert(0, "0")  # Default value

total_of_special_requests = tk.Label(root, text="Total of Special Requests:")
total_of_special_requests.grid(row=23, column=0)
entry_total_of_special_requests = tk.Entry(root)
entry_total_of_special_requests.grid(row=23, column=1)
entry_total_of_special_requests.insert(0, "0")  # Default value

days_in_waiting_list = tk.Label(root, text="Days in Waiting List:")
days_in_waiting_list.grid(row=24, column=0)
entry_days_in_waiting_list = tk.Entry(root)
entry_days_in_waiting_list.grid(row=24, column=1)
entry_days_in_waiting_list.insert(0, "0")  # Default value

booking_changes = tk.Label(root, text="Booking Changes:")
booking_changes.grid(row=25, column=0)
entry_booking_changes = tk.Entry(root)
entry_booking_changes.grid(row=25, column=1)
entry_booking_changes.insert(0, "0")  # Default value


# Create a predict button
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.grid(row=26, column=0, columnspan=2)

# Run the Tkinter event loop
root.mainloop()

