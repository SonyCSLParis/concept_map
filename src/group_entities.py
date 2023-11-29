"""
Group entities extracted for single file and for each subfolder
"""

import os


def get_unique_entities_from_files_in_subfolders(input_folder, output_folder):
    # Iterate through all subfolders in the main folder
    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        if os.path.isdir(subfolder_path):
            unique_entities = set()

            # Iterate through all files in the subfolder
            for root, dirs, files in os.walk(subfolder_path):
                for file_name in files:
                    if file_name.endswith(".txt"):
                        file_path = os.path.join(root, file_name)

                        # Open each file and extract unique entities
                        with open(file_path, "r") as file:
                            entities = file.readlines()
                            unique_entities.update(entities)

            output_root = root.replace(input_folder, output_folder)
            os.makedirs(output_root, exist_ok=True)

            output_file_name = file_name.replace(".txt", "-unique_entities.txt")
            output_file_path = os.path.join(output_root, output_file_name)

            with open(output_file_path, "w") as output_file:
                result_str = '\n'.join(unique_entities)
                output_file.write(result_str)


def set_entities(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith(".txt"):
                file_path = os.path.join(root, file_name)

                with open(file_path, "r") as file:
                    entities = file.readlines()
                    unique_entities = list(set(entities))
                    entities_without_newline = [entity.rstrip('\n') for entity in unique_entities]
                    # print(entities_without_newline)

                output_root = root.replace(input_folder, output_folder)
                os.makedirs(output_root, exist_ok=True)

                output_file_name = file_name.replace(".txt", "-preprocessed.txt")
                output_file_path = os.path.join(output_root, output_file_name)

                with open(output_file_path, "w") as output_file:
                    result_str = '\n'.join(entities_without_newline)
                    output_file.write(result_str)
