"""
Lead Month Performance Classifier

This script uses a Gaussian Naive Bayes classifier to predict whether a community's
lead pacing is Good, Watch, Bad, or Urgent based on mid-month performance.

How to run:
1. Make sure Python is installed.
2. Install required packages:
   pip install pandas scikit-learn
3. Run the file:
   python lead_month_performance_classifier.py
4. Enter the requested lead pacing inputs when prompted.
"""

import pandas as pd
from sklearn.naive_bayes import GaussianNB


def calculate_projected_eom(current_leads, day_of_month, total_days):
    """Calculate projected end of month leads."""
    return (current_leads / day_of_month) * total_days


def calculate_difference(projected_eom, previous_month_leads):
    """Calculate lead difference versus the previous month."""
    return projected_eom - previous_month_leads


def calculate_mom_change(difference, previous_month_leads):
    """Calculate month over month percentage change."""
    if previous_month_leads == 0:
        return 0
    return (difference / previous_month_leads) * 100


def assign_label(difference):
    """
    Assign performance label based on difference versus previous month.

    Good = Difference >= 0
    Watch = Difference between -1 and -10
    Bad = Difference between -11 and -20
    Urgent = Difference <= -21
    """
    if difference >= 0:
        return "Good"
    elif difference >= -10:
        return "Watch"
    elif difference >= -20:
        return "Bad"
    else:
        return "Urgent"


def build_training_data():
    """Create sample training data for the Naive Bayes classifier."""
    rows = [
        # previous_month_leads, current_leads, day_of_month, total_days
        [50, 30, 15, 30],
        [45, 23, 15, 30],
        [60, 29, 15, 30],
        [70, 35, 15, 30],
        [40, 22, 15, 30],
        [55, 26, 15, 30],
        [80, 38, 15, 30],
        [65, 30, 15, 30],
        [75, 32, 15, 30],
        [90, 35, 15, 30],
        [50, 18, 15, 30],
        [60, 20, 15, 30],
        [70, 22, 15, 30],
        [85, 25, 15, 30],
        [95, 28, 15, 30],
        [45, 10, 15, 30],
        [60, 12, 15, 30],
        [80, 18, 15, 30],
        [100, 20, 15, 30],
        [110, 24, 15, 30],
        [35, 18, 15, 30],
        [48, 24, 15, 30],
        [52, 19, 15, 30],
        [68, 27, 15, 30],
        [72, 26, 15, 30],
    ]

    data = []
    for previous_month, current, day, total_days in rows:
        projected = calculate_projected_eom(current, day, total_days)
        difference = calculate_difference(projected, previous_month)
        mom_change = calculate_mom_change(difference, previous_month)
        label = assign_label(difference)

        data.append([
            previous_month,
            current,
            day,
            total_days,
            projected,
            difference,
            mom_change,
            label
        ])

    columns = [
        "Previous Month Leads",
        "Current Leads",
        "Day of Month",
        "Total Days in Month",
        "Projected EOM Leads",
        "Difference vs Previous Month",
        "MoM Percent Change",
        "Label"
    ]

    return pd.DataFrame(data, columns=columns)


def train_model(df):
    """Train Gaussian Naive Bayes model."""
    feature_columns = [
        "Previous Month Leads",
        "Current Leads",
        "Day of Month",
        "Total Days in Month",
        "Projected EOM Leads",
        "Difference vs Previous Month",
        "MoM Percent Change"
    ]

    X = df[feature_columns]
    y = df["Label"]

    model = GaussianNB()
    model.fit(X, y)

    return model, feature_columns


def predict_performance(model, feature_columns):
    """Collect user input and display model prediction."""
    print("\nLead Month Performance Classifier")
    print("--------------------------------")

    community_name = input("Enter community name: ")

    previous_month_leads = float(input("Enter previous month leads: "))
    current_leads = float(input("Enter current leads so far this month: "))
    day_of_month = float(input("Enter current day of the month: "))
    total_days = float(input("Enter total number of days in the month: "))

    projected_eom = calculate_projected_eom(current_leads, day_of_month, total_days)
    difference = calculate_difference(projected_eom, previous_month_leads)
    mom_change = calculate_mom_change(difference, previous_month_leads)
    business_label = assign_label(difference)

    new_data = pd.DataFrame([[
        previous_month_leads,
        current_leads,
        day_of_month,
        total_days,
        projected_eom,
        difference,
        mom_change
    ]], columns=feature_columns)

    prediction = model.predict(new_data)[0]
    probabilities = model.predict_proba(new_data)[0]
    classes = model.classes_

    probability_dict = {
        class_name: probability * 100
        for class_name, probability in zip(classes, probabilities)
    }

    print("\nResults")
    print("-------")
    print(f"Community: {community_name}")
    print(f"Projected EOM Leads: {projected_eom:.0f}")
    print(f"Difference vs Previous Month: {difference:+.0f}")
    print(f"MoM Change: {mom_change:+.0f}%")
    print("\nPrediction Probabilities:")

    for label in ["Good", "Watch", "Bad", "Urgent"]:
        probability = probability_dict.get(label, 0)
        print(f"{label}: {probability:.0f} ({mom_change:+.0f}%)")

    print(f"\nNaive Bayes Prediction: {prediction}")
    print(f"Business Rule Label: {business_label}")

    if business_label == "Good":
        print("Recommendation: Community is pacing at or above last month.")
    elif business_label == "Watch":
        print("Recommendation: Community is slightly behind and should be monitored.")
    elif business_label == "Bad":
        print("Recommendation: Community is meaningfully behind and may need optimization.")
    else:
        print("Recommendation: Community is at high risk and needs immediate attention.")


def main():
    training_df = build_training_data()

    print("Sample Frequency Table")
    print("----------------------")
    print(training_df["Label"].value_counts().to_string())

    print("\nSample Training Data")
    print("--------------------")
    print(training_df.to_string(index=False))

    model, feature_columns = train_model(training_df)
    predict_performance(model, feature_columns)


if __name__ == "__main__":
    main()
