import os
import ast
import re
from settings import *


def transform_content(input_content):
    results = []
    for item in input_content:
        regex = r"('head'|'type'|'tail')"
        subst = ""
        result = re.sub(regex, subst, item, 0, re.MULTILINE)
        regex_2 = r"(:|\[\{|\s{2,})"
        result_2 = re.sub(regex_2, subst, result, 0, re.MULTILINE)
        regex_3 = r"\[\{"
        subst_2= "\\n"
        result_3 = re.sub(regex_3, subst_2, result_2, 0, re.MULTILINE)
        results.append(result_3)
    return results

def save_to_file(output_text, output_file_path):
    with open(output_file_path, 'w') as output_file:
        output_text = '\n'.join(output_text)
        output_file.write(output_text)

def process_output_file(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith(".txt"):
                file_path = os.path.join(root, file_name)
                with open(file_path, "r") as file:
                    input_lines = file.read()
                    matches = re.findall(r'\{.*?\}', input_lines)
                    input_data_list = [f"[{match}]" for match in matches]
                    output_data = transform_content(input_data_list)
                    output_root = root.replace(input_folder, output_folder)
                    os.makedirs(output_root, exist_ok=True)
                    output_file_name = file_name.replace(".txt", "-transformed.txt")  # Change the extension to .txt
                    output_file_path = os.path.join(output_root, output_file_name)
                    save_to_file(output_data, output_file_path)
                    print(f'Transformed output saved to: {output_file_path}')


if __name__ == '__main__':
    input_folder = OUTPUT_TRIPLETS_FINE_TUNE
    output_folder = OUTPUT_TRIPLETS_PREPORCESSED
    process_output_file(input_folder, output_folder)
