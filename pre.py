import pandas as pd
import numpy as np
import os
import json
import random
import csv
from sklearn.linear_model import LinearRegression

# Hardcoded file path (change this to your actual file path)
FILE_PATH = r"C:\Users\haris\Desktop\linear_regression\uploads\Sales_Data_for_Analysis.tsv"  # Example: "C:/Users/YourName/Documents/data.csv"
OUTPUT_FILE = "output.txt"

# Function to detect delimiter
def detect_delimiter(file_path):
    """Detects the delimiter in a given CSV/TSV file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        sample = f.read(1024)
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(sample).delimiter
        return delimiter

# Function to process dataframe
def process_dataframe(df):
    try:
        # Data Cleaning
        df.columns = df.columns.str.strip()

        # Rename columns for consistency
        df.rename(columns={
            "PERIOD": "year", 
            "QTY": "Quantity", 
            "TOTAL PRICE (INR)": "Item Total",
            "CURRENCY": "Currency", 
            "EX RATE": "Exchange Rate", 
            "PART NO": "PART NO"
        }, inplace=True)

        df.dropna(subset=["PART NO"], inplace=True)
        df["year"] = pd.to_datetime(df["year"], errors="coerce", dayfirst=True).dt.year
        df.dropna(subset=["year"], inplace=True)
        df["year"] = df["year"].astype(int)
        df["Currency"] = df["Currency"].str.strip().str.upper()

        # Fill missing exchange rates with default value
        df["Exchange Rate"] = df["Exchange Rate"].fillna(75)

        # Convert USD to INR
        df.loc[df["Currency"] == "USD", "Item Total"] *= df["Exchange Rate"]
        df["Currency"] = "INR"

        # Aggregate data
        latest_year = df["year"].max()
        grouped = df.groupby(["PART NO", "year"])[["Quantity", "Item Total"]].sum().reset_index()

        predictions = []
        customer_names = ["CustomerA", "CustomerB", "CustomerC", "CustomerD", "CustomerE"]

        for part_no in grouped["PART NO"].unique():
            part_data = grouped[grouped["PART NO"] == part_no].copy()
            existing_customer = random.choice(customer_names)

            if len(part_data) < 3:
                pred_quantity = int(round(part_data["Quantity"].mean())) if len(part_data) > 1 else 0
                pred_total = int(round(part_data["Item Total"].mean())) if len(part_data) > 1 else 0
                predictions.append({
                    "PART NO": part_no, "year": latest_year + 1, "Predicted Quantity": max(0, pred_quantity),
                    "Predicted Item Total": max(0, pred_total), "Min Quantity": 0, "Max Quantity": 0, 
                    "Min Item Total": 0, "Max Item Total": 0, "Currency": "INR", 
                    "Customer Name": existing_customer, "Quarter": "Q4"
                })
                continue

            try:
                # Linear Regression for Quantity
                model_quantity = LinearRegression().fit(part_data[["year"]], part_data["Quantity"])
                predicted_year = pd.DataFrame([[latest_year + 1]], columns=["year"])
                pred_quantity = max(0, int(round(model_quantity.predict(predicted_year)[0])))

                # Linear Regression for Item Total
                model_total = LinearRegression().fit(part_data[["year"]], part_data["Item Total"])
                pred_total = max(0, int(round(model_total.predict(predicted_year)[0])))

                residual_quantity = np.abs(part_data["Quantity"] - model_quantity.predict(part_data[["year"]])).mean()
                residual_total = np.abs(part_data["Item Total"] - model_total.predict(part_data[["year"]])).mean()

                min_quantity, max_quantity = int(max(0, pred_quantity - residual_quantity)), int(max(0, pred_quantity + residual_quantity))
                min_total, max_total = int(max(0, pred_total - residual_total)), int(max(0, pred_total + residual_total))

            except Exception as e:
                print(f"Error in regression for {part_no}: {e}")
                pred_quantity, pred_total = max(0, int(part_data["Quantity"].mean())), max(0, int(part_data["Item Total"].mean()))
                min_quantity, max_quantity, min_total, max_total = pred_quantity, pred_quantity, pred_total, pred_total

            predictions.append({
                "PART NO": part_no, "year": latest_year + 1, "Predicted Quantity": pred_quantity,
                "Predicted Item Total": pred_total, "Min Quantity": min_quantity, "Max Quantity": max_quantity,
                "Min Item Total": min_total, "Max Item Total": max_total, "Currency": "INR",
                "Customer Name": existing_customer, "Quarter": "Q4"
            })

        # Convert numpy.int64 to Python int
        for item in predictions:
            for key, value in item.items():
                if isinstance(value, np.integer):
                    item[key] = int(value)

        return predictions

    except Exception as e:
        print(f"Error processing dataframe: {e}")
        return None

# Main script execution
if __name__ == "__main__":
    if not os.path.exists(FILE_PATH):
        print(f"Error: File '{FILE_PATH}' not found.")
    else:
        try:
            # Detect delimiter and read CSV/TSV correctly
            delimiter = detect_delimiter(FILE_PATH)
            df = pd.read_csv(FILE_PATH, sep=delimiter, engine='python', on_bad_lines='skip')

            # Process the dataframe
            predictions = process_dataframe(df)
            if predictions is None:
                print("Error: Failed to process data.")
            else:
                # Save predictions to a file
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(predictions, f, indent=4)
                
                print(f"Predictions saved to {OUTPUT_FILE}")

        except Exception as e:
            print(f"Error reading file: {e}")
