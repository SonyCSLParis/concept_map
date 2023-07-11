import os

from settings import *


def get_token_from_index(doc, index):
    token = doc[index]

    if type(token)==str:
        return token
    else :
        return token.text
    # except IndexError:
    #     return None


def append_if_not_found(list_of_lists, value):
    new_list_of_lists = []
    for inner_list in list_of_lists:
        new_inner_list = []
        int_v = int(value)
        for item in inner_list:
            if int_v != item:
                new_inner_list.append(item)
        new_list_of_lists.append(new_inner_list)
    return new_list_of_lists



def substitute_tokens(list_of_lists, doc, index):
    substituted_tokens = []
    for inner_list in list_of_lists:
        substituted_inner_list = []
        for idx in inner_list:
            if 0 <= idx < len(doc):
                substituted_inner_list.append(doc[idx].text)
            else:
                substituted_inner_list.append(None)
        substituted_tokens.append(substituted_inner_list)

    if 0 <= index < len(doc):
        token_to_substitute = doc[index]
    else:
        token_to_substitute = None

    new_tokens = []
    for i, token in enumerate(doc):
        if token_to_substitute is not None and any(i in sublist for sublist in list_of_lists):
            new_tokens.append(token_to_substitute.text)
        else:
            new_tokens.append(token.text)

    new_text = ' '.join(new_tokens)
    # new_doc = spacy.tokens.Doc(doc.vocab, words=new_text)

    return new_text


def perform_coreference_resolution(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    nlp.add_pipe('coreferee')

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)

            with open(file_path, 'r') as file:
                text = file.read()

            ## Perform coreference resolution
            doc = nlp(text)
            print(doc._.coref_chains.print())
            x = doc._.coref_chains
            if len(doc._.coref_chains) > 0:
                for chain in doc._.coref_chains:
                    index = chain.most_specific_mention_index
                    token = get_token_from_index(doc, index)
                    if token:
                        print(f"Token at index {index}: {token}")
                        list_chains = chain.mentions
                        items = append_if_not_found(list_chains, index)
                        print(items)
                        doc = substitute_tokens(items, doc, index)

            output_file_path = os.path.join(output_folder, filename)

            with open(output_file_path, 'w') as file:
                file.write(doc.text)

            print(f"Coreference resolution completed for {filename}. Output saved to {output_file_path}")

# def perform_coreference_resolution(input_folder, output_folder):
#     # Create output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # Initialize coreference resolver
#     nlp.add_pipe('coreferee')
#
#     # Iterate over files in the input folder
#     for filename in os.listdir(input_folder):
#         if filename.endswith(".txt"):
#             file_path = os.path.join(input_folder, filename)
#
#             # Read the contents of the input file
#             with open(file_path, 'r') as file:
#                 lines = file.readlines()
#
#             # Perform coreference resolution for each line
#             resolved_lines = []
#             for line in lines:
#                 # Split the line by whitespace
#                 doc = nlp(line)
#
#                 elements = line.strip().split()
#
#                 # Perform coreference resolution on the first element
#                 coref_result = doc._.coref_chains.resolve(doc[0])
#
#                 # If coreference is found, modify the line accordingly
#                 print(coref_result)
#                 if coref_result:
#                     coreferee = coref_result[0]['entity']
#                     elements.insert(1, "coref")
#                     elements.insert(0, coreferee)
#
#                 # Append the modified line to the resolved lines list
#                 resolved_lines.append(' '.join(elements))
#
#             # Create the output file path
#             output_file_path = os.path.join(output_folder, filename)
#
#             # Write the resolved lines to the output file
#             with open(output_file_path, 'w') as file:
#                 file.writelines(resolved_lines)
#
#             print(f"Coreference resolution completed for {filename}. Output saved to {output_file_path}")
