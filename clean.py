import pandas as pd
import os

def clean_and_update_dehum_data(input_dir='csvs'):
    output_file = os.path.join(input_dir, 'dehums.csv')

    dehum_file = None
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv'):
            if 'Dehum' in file_name or 'EnPI' in file_name:
                dehum_file = os.path.join(input_dir, file_name)
                break

    if dehum_file is None:
        print("No dehumidifier file found.")
        return

    # Read and clean dehums data
    dehums = pd.read_csv(dehum_file)
    dehums.fillna(0, inplace=True)
    dehums['Timestamp'] = pd.to_datetime(dehums['Timestamp'])
    dehums['Date'] = dehums['Timestamp'].dt.date
    dehums['Time'] = dehums['Timestamp'].dt.time

    # Add Year, Month, Week, Day columns for dashboard compatibility
    dehums['Year'] = dehums['Timestamp'].dt.year.astype(str)
    dehums['Month'] = dehums['Timestamp'].dt.strftime('%b')
    dehums['Week'] = dehums['Timestamp'].dt.isocalendar().week.astype(int)
    dehums['Day'] = dehums['Timestamp'].dt.strftime('%a')

    # Output detailed file (like steam.csv), but drop Timestamp
    detailed_cols = ['Date', 'Time', 'Year', 'Month', 'Week', 'Day', 'VG150 Pack. PBL01', 'VG221 Pack. PPL02']
    for col in detailed_cols:
        if col not in dehums.columns:
            dehums[col] = 0
    combined = dehums[detailed_cols]

    # If output file exists, append new rows (without duplicates)
    if os.path.exists(output_file):
        existing = pd.read_csv(output_file)
        all_data = pd.concat([existing, combined], ignore_index=True)
        all_data.drop_duplicates(subset=['Date', 'Time', 'VG150 Pack. PBL01', 'VG221 Pack. PPL02'], inplace=True)
        all_data.sort_values(['Date', 'Time'], inplace=True)
        all_data.to_csv(output_file, index=False)
        print(f"Appended new data to existing dehum file: {output_file}")
    else:
        combined.to_csv(output_file, index=False)
        print(f"Detailed dehum file saved: {output_file}")
    return combined

if __name__ == "__main__":
    clean_and_update_dehum_data()