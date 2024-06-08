import pandas as pd
import pickle


def flatten_code_list(code_list):
    """
    Flatten a nested list of integers into a single list.
    """
    flat_list = []
    for sublist in code_list:
        if isinstance(sublist, list):
            flat_list.extend(flatten_code_list(sublist))
        else:
            flat_list.append(sublist)
    return flat_list


def pickle_to_excel(pickle_file, excel_file):
    try:
        # Load data from pickle file
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)

        # Filter out rows with empty code lists
        data = data[data["code"].apply(len) > 0]

        # Flatten the nested code lists
        data["code"] = data["code"].apply(flatten_code_list)

        # Extract labels and codes from the filtered data
        labels = data["label"]
        codes = data["code"]

        # Create a DataFrame with "label" and "code" columns
        df = pd.DataFrame({"label": labels, "code": codes})

        # Save DataFrame to Excel file
        df.to_excel(excel_file, index=False)

        print("Data saved to Excel file successfully.")
    except Exception as e:
        print("An error occurred:", str(e))


# Usage example
pickle_to_excel("test_block.pkl", "test_block.xlsx")
