import pandas as pd

def load_reviews(file_path):
    return pd.read_csv(file_path, delimiter=";")

def load_reference_translations(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]
