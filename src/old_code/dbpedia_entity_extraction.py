# -*- coding: utf-8 -*-
"""
Running spacy pipeline on data with DBpedia Spotlight

Before running this script: install dbpedia spotlight locally and start the server
To run the server locally: java -Xmx8G -jar /path/to/dbpedia-spotlight/rest-1.1-jar-with-dependencies.jar \
     en http://localhost:2222/rest
"""
import os
from settings import nlp

def extract_dbpedia_entities(input_folder: str, output_folder: str,
                             confidence: float = 0.7):
    """ Retrieving DBpedia entities in a .txt file.
    Template
    {surface form}\t{dbpedia IRI}\t{offset}\t{similarity score}\n """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    nlp.add_pipe("dbpedia_spotlight",
             config={
                'confidence': confidence,
                'dbpedia_rest_endpoint': 'http://localhost:2222/rest'})

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            # Retrieve DBpedia entities
            doc = nlp(text)
            db_entities = [ent._.dbpedia_raw_result for ent in doc.ents if ent._.dbpedia_raw_result]

            output_file_path = os.path.join(output_folder, filename)
            f_output = open(output_file_path, "w", encoding='utf-8')
            f_output.write('\n'.join(
                [f"{ent['@surfaceForm']}\t{ent['@URI']}\t{ent['@offset']}\t{ent['@similarityScore']}" \
                    for ent in db_entities]
            ))
            f_output.close()

            print(f"DBpedia entity extraction completed for {filename}. Output saved to {output_file_path}")