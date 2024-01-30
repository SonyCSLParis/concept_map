"""
Group entities extracted for single file and for each subfolder
"""
import os

class EntityGrouper:
    """ Grouping entities across files """
    def __call__(self, entities):
        """ Set of entities """
        entities = list(set(entities))
        return [x.rstrip('\n') for x in entities]

    def get_set_single_folder(self, input_folder: str, output_folder: str):
        """ Extract single entities from file """
        for root, _, files in os.walk(input_folder):
            for file_name in [x for x in files if x.endswith(".txt")]:
                file_path = os.path.join(root, file_name)

                with open(file_path, "r", encoding='utf-8') as file:
                    entities = file.readlines()
                    entities = self(entities=entities)

                output_root = root.replace(input_folder, output_folder)
                os.makedirs(output_root, exist_ok=True)

                output_file_name = file_name.replace(".txt", "-preprocessed.txt")
                output_file_path = os.path.join(output_root, output_file_name)

                with open(output_file_path, "w", encoding='utf-8') as output_file:
                    result_str = '\n'.join(entities)
                    output_file.write(result_str)

    @staticmethod
    def get_set_multi_folder(input_folder: str, output_folder: str):
        """ Get unique entities from files in subfolders """
        # Iterate through all subfolders in the main folder
        for subfolder in os.listdir(input_folder):
            subfolder_path = os.path.join(input_folder, subfolder)
            if os.path.isdir(subfolder_path):
                unique_entities = set()

                # Iterate through all files in the subfolder
                for root, _, files in os.walk(subfolder_path):
                    for file_name in files:
                        if file_name.endswith(".txt"):
                            # Open each file and extract unique entities
                            with open(os.path.join(root, file_name), "r", encoding='utf-8') as file:
                                entities = file.readlines()
                                unique_entities.update(entities)

                        output_root = root.replace(input_folder, output_folder)
                        os.makedirs(output_root, exist_ok=True)

                        output_file_name = file_name.replace(".txt", "-unique_entities.txt")
                        output_file_path = os.path.join(output_root, output_file_name)

                        with open(output_file_path, "w", encoding='utf-8') as output_file:
                            result_str = '\n'.join(unique_entities)
                            output_file.write(result_str)

    def main_folder(self, input_folder: str, inter_folder: str, output_folder: str):
        """ Main """
        self.get_set_multi_folder(input_folder=input_folder, output_folder=inter_folder)
        self.get_set_multi_folder(input_folder=inter_folder, output_folder=output_folder)
