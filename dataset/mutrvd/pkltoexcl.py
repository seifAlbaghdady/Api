import pandas as pd
import pickle


def pickle_to_excel(pickle_file, excel_file):
    try:
        # Load data from pickle file
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)

        # Check the number of rows in the loaded data
        num_rows = len(data)
        print("Number of rows in loaded data:", num_rows)

        # Check if data is a list
        if not isinstance(data, list):
            raise ValueError("Data is not a list.")

        # Check if each item in data is a dictionary with 'label' and 'code' keys
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("Each item in data must be a dictionary.")
            if "label" not in item or "code" not in item:
                raise ValueError(
                    "Each dictionary in data must have 'label' and 'code' keys."
                )

        # Extract labels and codes from the data
        labels = [item["label"] for item in data]
        codes = [item["code"] for item in data]

        # Create a DataFrame with "label" and "code" columns
        df = pd.DataFrame({"label": labels, "code": codes})

        # Save DataFrame to Excel file
        df.to_excel(excel_file, index=False)

        print("Data saved to Excel file successfully.")
    except Exception as e:
        print("An error occurred:", str(e))


# Usage example
pickle_to_excel("train.pkl", "train.xlsx")
