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

# def combine_triplets_to_joint_summary(parent_folder_path, output_folder_path):
#     for file in os.listdir(parent_folder_path):
#         file_number = file.split("-")[0]
#         output_file_name = f"{file_number}-joint_summary.txt"
#
#         output_file_path = os.path.join(output_folder_path, output_file_name)
#         input_file_path = os.path.join(parent_folder_path, file)
#         with open(output_file_path, "w") as output_file:
#             if file.startswith(file_number):
#                 file_path = os.path.join(parent_folder_path, input_file_path)
#                 with open(file_path, "r") as file:
#                     text = file.read()
#                     output_file.write(text)
#
