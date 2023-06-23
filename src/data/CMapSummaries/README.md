
# CMapSummaries
v1.0, 2017-01-31


CMapSummaries is a multi-document summarization corpus with concept maps as summaries. It contains 30 document clusters with around 40 web documents on educational topics, each with a reference concept map summarizing it.

It was introduced in the following publication:

```bibtex
@inproceedings{
	title = {Bringing Structure into Summaries: Crowdsourcing a Benchmark Corpus of Concept Maps},
	author = {Falke, Tobias and Gurevych, Iryna},
	booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
	pages = {to appear},
	year = {2017},
	address = {Copenhagen, Denmark}
}
```

The corpus is available at the folling website:

[https://www.ukp.tu-darmstadt.de/data/summarization/concept-map-summaries](https://www.ukp.tu-darmstadt.de/data/summarization/concept-map-summaries)

Corresponding software can be found at:

[https://github.com/UKPLab/emnlp2017-cmapsum-corpus](https://github.com/UKPLab/emnlp2017-cmapsum-corpus)

For questions, please contact Tobias Falke (lastname@aiphes.tu-darmstadt.de)

## Contents

The corpus is split into a training and test set. For each, a folder per topic contains the source documents and the summary concept map.

* **CMapSummaries**
	* **train**
		* topics.tsv
			* descriptions of topics, tab-seperated list
		* `<topic>`
			* `<topic>`.cmap 
				* summary concept map
			* `<topic>`-`<doc>`.txt 
				* source documents, one file per document
	* **test**
		* *same as training data*

*Please note:* plots of the concept maps (*.png) are included for convenience, but were created automatically with graphviz and tend to be quite messy.

A second folder contains supplementary material that is not necessary to perform and evaluate the summarization task, but gives additional insight into the creation of the corpus.

* **SupplementaryMaterial**
	* annotations.tsv
		* tab-seperated list of all propositions that were crowdsourced (see 4.4 in paper), containing the importance estimations obtained as well as information to trace the proposition back to the source documents.
	* AnnotationGuideline_*
		* guidelines used during corpus creation steps that involved expert annotations.
	* AMT\_HITExample_*
		* examples showing the HIT interface used on Mechanical Turk.


## Format

All text files are UTF-8 encoded. Summary concept maps (*.cmap) are provided as text files, with one proposition per line and the concepts and relation phrase being seperated by a tab. In other words, every line in the file represents an edge in the graph, labeled by the second column, and the set of unique labels in the first and third columns represent the set of nodes in the graph.


## License

* The annotations are licensed under [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode).
* The original content from the DIP 2016 Corpus keeps its original license.
* Please cite our paper if you use the data in your work.

