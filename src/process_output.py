import os
import ast
import re
from settings import *

def preprocess_input_text(input_lines):
    # Concatenate the list of strings into a single string
    input_text = ''.join(input_lines)

    # Remove double quotes around each triplet string
    input_text = re.sub(r'"({.*?})"', r'\1', input_text)
    return input_text

def transform_output(input_text):
    # Use ast.literal_eval to safely convert the string to a Python object
    triplets = ast.literal_eval(input_text)

    # Transform triplets into the desired format
    transformed_text = '\n'.join([f"{triplet['head']}, {triplet['type']}, {triplet['tail']}" for triplet in triplets])

    return transformed_text

def save_to_file(output_text, output_file_path):
    with open(output_file_path, 'w') as output_file:
        output_file.write(output_text)

def process_output_file(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith(".txt"):
                file_path = os.path.join(root, file_name)
                with open(file_path, "r") as file:
                    input_lines = file.readlines()

                    # Preprocess the input text
                    input_text = preprocess_input_text(input_lines)

                    # Transform the preprocessed text
                    transformed_output = transform_output(input_text)

                    output_root = root.replace(input_folder, output_folder)
                    os.makedirs(output_root, exist_ok=True)
                    output_file_name = file_name.replace(".txt", "-transformed.txt")  # Change the extension to .txt
                    output_file_path = os.path.join(output_root, output_file_name)
                    save_to_file(transformed_output, output_file_path)
                    print(f'Transformed output saved to: {output_file_path}')


if __name__ == '__main__':
    input_folder = OUTPUT_TRIPLETS_FINE_TUNE
    output_folder = OUTPUT_TRIPLETS_PREPORCESSED
    process_output_file(input_folder, output_folder)
