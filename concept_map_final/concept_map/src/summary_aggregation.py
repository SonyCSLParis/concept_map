import os


def combine_triplets_to_joint_summary(parent_folder_path, output_folder_path):
    for folder_name in os.listdir(parent_folder_path):
        folder_path = os.path.join(parent_folder_path, folder_name)
        if os.path.isdir(folder_path):
            output_file_name = folder_name + "-joint_summary.txt"
            output_file_path = os.path.join(output_folder_path, output_file_name)

            with open(output_file_path, "w") as output_file:
                for file_name in os.listdir(folder_path):
                    if file_name.endswith("-triplets.txt"):
                        file_path = os.path.join(folder_path, file_name)
                        with open(file_path, "r") as file:
                            text = file.read()
                            output_file.write(text)

