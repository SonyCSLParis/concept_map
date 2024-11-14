"""
Global variables for the module
"""
# from spacy.pipeline.entity_linker import DEFAULT_NEL_MODEL
import spacy

nlp = spacy.load("en_core_web_sm", exclude=["static_vectors"])
nlp.add_pipe("entityLinker", last=True)

# nlp = spacy.load("en_core_web_lg")
