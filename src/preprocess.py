"""
Running spacy pipeline for preprocessing : remove stopwords,punctuation, double spaces and citations
"""

import os
import re
import spacy

nlp = spacy.load("en_core_web_sm")

def remove_patterns(text, patterns):
    for pattern in patterns:
        text = re.sub(pattern, '', text)
    return text

def find_index(doc, keywords):
    for keyword in keywords:
        index = next((i for i, token in enumerate(doc) if token.text.lower() == keyword.lower()), None)
        if index is not None:
            return index
    return None

def find_two_token_index(doc, two_tokens):
    for i in range(len(doc) - 1):
        if doc[i].text.lower() == two_tokens[0].lower() and doc[i + 1].text.lower() == two_tokens[1].lower():
            return i
    return None

def create_new_doc(doc, end_index):
    if end_index is not None:
        return doc[:end_index]
    else:
        return doc

def preprocess_text(text):
    patterns_to_remove = [
        r'Manuscripts, Medieval--Ireland.*?Trinity College Library, Dublin\.',
        r'\[Accessed[^\]]+\]',
        r'Slide \d+',
        r'http\S+',
        r'ß\?\?\?t\?\?\?Saßß?t\?\?\?st\?\?\?a\?\?;'
    ]

    text_without_patterns = remove_patterns(text, patterns_to_remove)
    doc = nlp(text_without_patterns)

    tokens_to_filter = [
        "Note:", "Please be patient until they appear.",
        "Illustration for Alfred Noyes' poem \"A Spell for a Fairy\" in Princess Mary's Gift Book by Claude Shepperson.",
        "Architects & Engineers for 9/11 Truth",
        "Reward: Elusive \"History's Business\" Episodewith Larry SilversteinAE911",
        "Follow Scientific American on Twitter", "@SciAm", "@SciamBlogs",
        "Visit Scientific American.com for the latest in science, health and technology news.",
        "© 2011 Scientific American.com. All rights reserved.", "Edit by Thomas Koitzsch",
        "edit", "For more information, visit www.nationaltrust.org.uk/bodiam-castleor call 01580 831324.",
        "Washington-Centerville Public Library,111 West Spring Valley Road,Centerville, OH 45458,www.wclibrary.info",
        "Information on the facsimile from the company that made the facsimile (In Dutch)",
        "By John L. Cisne Perception, Vol.38 (2009)",
        "[Interview]",
        "Gallery",
        "Website:",
        "Go next[edit]",
        "edit",
        "[]",
        "[read less]",
        "Thank you for listening!",
        "Newspaper Article",
        "From Wikipedia, the free encyclopedia",
        "For more information visit:",
        "Image ID:124398484Copyright: Iryna Rasko Available in high-resolution and several sizes to fit the needs of your project."
    ]

    filtered_tokens = [token.text for token in doc if token.text not in tokens_to_filter]
    filtered_doc = nlp(" ".join(filtered_tokens))

    keywords_to_find = ["contributors", "abbreviations", "sources", "refs", "visited", "link", "references", "source",
                        "directions", "sources", "refs", "references", "ReferencesCole", "bibliography",
                        "BibliographyArmitage", "copyright", "notes", "note", "notes1"]

    for keyword in keywords_to_find:
        keyword_index = find_index(filtered_doc, [keyword])

        if keyword_index is not None:
            new_doc = create_new_doc(doc, keyword_index)
            new_text = new_doc.text
            print(f"{keyword.capitalize()} found. New document:")
            print(new_text)
        else:
            new_doc = doc
            new_text = new_doc.text
            print(f"{keyword.capitalize()} not found in the text.")

    two_token_keywords_to_find = [["more", "information"], ["further", "information"],["see", "also"], ["about", "the","water"],["go", "to","text"]]

    for keywords in two_token_keywords_to_find:
        keyword_index = find_two_token_index(doc, keywords)

        if keyword_index is not None:
            new_doc = create_new_doc(doc, keyword_index)
            new_text = new_doc.text
            print(f"{' '.join(keywords).capitalize()} found. New document:")
            print(new_text)
        else:
            new_doc = doc
            new_text = new_doc.text
            print(f"{' '.join(keywords).capitalize()} not found in the text.")

    cleaned_text = re.sub(r'\s+', ' ', new_doc.text).strip()
    cleaned_text = re.sub(r'\[\d+\]\[\d+\]\[\d+\]', '', cleaned_text)

    return cleaned_text

def preprocess_folder(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith(".txt"):
                file_path = os.path.join(root, file_name)

                with open(file_path, "r") as file:
                    text = file.read()
                    doc = nlp(text)
                    sentences = [sent.text.strip() for sent in doc.sents]

                    cleaned_text = [preprocess_text(sent) for sent in sentences]
                    processed_text = '\n'.join(cleaned_text)

                output_root = root.replace(input_folder, output_folder)
                os.makedirs(output_root, exist_ok=True)

                output_file_name = file_name.replace(".txt", "-preprocessed.txt")
                output_file_path = os.path.join(output_root, output_file_name)

                with open(output_file_path, "w") as output_file:
                    output_file.write(processed_text)

def preprocess_bio(input_file_path,output_file_path):
    with open(input_file_path, "r") as file:
        text = file.read()
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        cleaned_text = [preprocess_text(sent) for sent in sentences]
        processed_text = '\n'.join(cleaned_text)

        with open(output_file_path, "w") as output_file:
            output_file.write(processed_text)