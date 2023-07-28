import os


def merge_files_by_prefix(directory_path, output_folder):
    file_dict = {}

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        if os.path.isfile(file_path) and filename.endswith('.txt'):
            prefix = filename.split('-')[0]

            if prefix in file_dict:
                with open(file_path, 'r') as file:
                    file_dict[prefix].append(file.read())
            else:
                with open(file_path, 'r') as file:
                    file_dict[prefix] = [file.read()]

    os.makedirs(output_folder, exist_ok=True)

    for prefix, content_list in file_dict.items():
        merged_content = '\n'.join(content_list)

        output_filename = f'{prefix}-merged.txt'
        output_path = os.path.join(output_folder, output_filename)

        with open(output_path, 'w') as file:
            file.write(merged_content)