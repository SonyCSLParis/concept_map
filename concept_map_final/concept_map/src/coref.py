import spacy
from settings import *

def perform_coreference_resolution(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize coreference resolver
    nlp.add_pipe('coreferee')

    # Iterate over files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)

            # Read the contents of the input file
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Perform coreference resolution for each line
            resolved_lines = []
            for line in lines:
                # Split the line by whitespace
                doc = nlp(line)

                elements = line.strip().split()

                # Perform coreference resolution on the first element
                coref_result = doc._.coref_chains.resolve(doc[0])

                # If coreference is found, modify the line accordingly
                print(coref_result)
                if coref_result:
                    coreferee = coref_result[0]['entity']
                    elements.insert(1, "coref")
                    elements.insert(0, coreferee)

                # Append the modified line to the resolved lines list
                resolved_lines.append(' '.join(elements))

            # Create the output file path
            output_file_path = os.path.join(output_folder, filename)

            # Write the resolved lines to the output file
            with open(output_file_path, 'w') as file:
                file.writelines(resolved_lines)

            print(f"Coreference resolution completed for {filename}. Output saved to {output_file_path}")
