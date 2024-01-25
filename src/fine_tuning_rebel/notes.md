

### REBEL fine-tuning

Originally pretrained on **relation extraction** task.

**Core idea** = fine-tune the model on concept map triple extraction. Can the model learn with few samples?
**Potential cons:** vocabulary is usually smaller/constrained in these tasks, more open in concept maps.

* `Corpora`: Falke's dataset

* `extract_predicates.py`
Analyse predicates from concept maps (to see if any more frequent than others, types, etc).
Can be run on the following two folders:
    * `all_gs_multi`: all concept maps from multi-document summarization
    * `all_gs_single`: all concept maps from single-document summarization
Also save the following file containing the predicate labels:
    * `predicate_label.txt`

* `map_triple_sentence.py`
For single-document summarisation, attempting to extract original sentence from which the triple was extracted. Only for Biology dataset (only one that was usable).
Rule-based system: mapping only if the subject and object are in the sentence, and if all the verbs' lemmas in the predicate are also in the sentence. Save the following document:
    * `cm_biology.csv`

* `divide_train_eval_test.py`
Divide the context sentences with their triples (`cm_biology.csv`) into train/eval/test set. Output files:
    * `cm_biology_eval.csv`
    * `cm_biology_test.csv`
    * `cm_biology_train.csv`

* `extract_vocab.py`
Extract vocab from training data (necessary for fine-tuning rebel)


* `rebel_fine_tuning.py`
Fine-tune REBEL model. Save the model:
    * `finetuned_rebel.pth`


* `run_rebel.py`
Self-explanatory.

---


### Rule-based triple extraction

**Core idea** = using dependency tree to extract triples from text (linking NPs via dependency paths that contain verbs)
**Potential cons:** cumbersome, overfitting

* `rule_based.py`
Extract triples using a rule-based system

---

### Pipeline

Changes to be made
* Instead of applying spacy nlp in each function, do this in the beginning (and pass the doc as argument)
* Saving file in the classes, functions should only perform the transformations
* some helper file with re-used functions
* what to do with settings.py? -> probably overwritten when running each pipeline independently

Classes
* ConceptMap
    * generate visual graph from text 
        * input = lines of text
        * output = Network graph
* CoreferenceResolution
    * main
        * input = text 
        * output = string/spacy doc
* EntityExtraction
    * DBpedia entity extraction (figure out how to use it)
        * input = text
        * output = list of entities
* Metrics
    * main
        * input = list of text 
        * output = csv
    * Falke's metrics (check the differences)
        * input = list of text
        * output = csv
* TriplesExtraction
    * REBEL model
        * input = text
        * output = list of triples
    * Rule-based model
        * Martina's orig
        * Ines' with NPs+dependency paths
    * Aggregate triplets
* ImportanceRanking
    * main
        * input = list of sentences
        * output =  ranked sentences (top k)
* PreProcessor
    * main
        * input = text
        * output = text
* Summarizer
    * summary aggregation (check further when cleaned)

    * summary generation (check further when cleaned)
* Pipeline
    * when calling the pipeline, creating a folder with
        * sub-folders (storing each step of the pipeline)
        * config or similar file: which steps of the pipeline + params
    * Minimal step = triples extraction
    * Possible steps of the pipeline + params (in order)
        * Summary: GPT || Symbolic (textrank?)
        * Preprocessing: yes || no 
        * Coreference resolution: yes || no 
        * Importance ranking: yes || no
        * Triples extraction: rebel || dependency (filtering/merging boolean) || rebel + combination
        * Visual graph generation: yes || no
        * Metrics: yes  || no
    * Type: train/test (only for name of folder output), mds/sds
    * Input = folder of subfolders with .txt files to generate concept maps from
    * Output = Folder with all results


```python
class ConceptMapGenerationPipeline(
    type_generated="mds",
    type_data="train",
    summariser="gpt",
    pre_processing=True,
    coref=True,
    ranking=True,
    triples="rebel,dependency",
    triples_filtering=True,
    triples_merging=True,
    visual_graph=False,
    metrics=True
)
```

Parameters:
* `type_generated: {'mds', 'sds'}, default='mds'`
Multi-document summarisation vs. single-document summarisation
* `type_data: {'train', 'test'}, default='train'`
Type of data used (for output folder name)
* `summariser: {'gpt', 'textrank', 'none'}, default='gpt'`
Summary method, if any
* `pre_processing: bool, default=True`
* `coref: bool, default=True`
* `ranking: bool, default=True`
* `triples: {'rebel', 'dependency', 'rebel,dependency'}, default='rebel,dependency'`
Method for triple extraction
* `triples_filtering: bool, default=True`
Only if `dependency` in `triples, default=True` param
* `triples_merging: bool, default=True`
Only if `dependency` in `triples` param
* `visual_graph: bool, default=False`
* `metrics: bool, default=True`