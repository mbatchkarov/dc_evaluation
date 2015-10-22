# Evaluation via document classification
[![Build Status](https://travis-ci.org/mbatchkarov/dc_evaluation.svg?branch=master)](https://travis-ci.org/mbatchkarov/dc_evaluation)
[![Coverage Status](https://coveralls.io/repos/mbatchkarov/dc_evaluation/badge.svg?branch=master&service=github)](https://coveralls.io/github/mbatchkarov/dc_evaluation?branch=master)

## Prerequisites
	
 - Python 3.2+ (legacy Python not supported)
 - standard Python scientific stack (`numpy`, `scipy`, `scikit-learn`, etc)
 - our [utility library](https://github.com/mbatchkarov/DiscoUtils)
 - Stanford CoreNLP (tested with version `3.5.2` from 2015-04-20)
 - sample corpus of labelled documents located at `data/web`
 - pre-computed vectors for words and phrases of interest. This distribution includes randomly generated vectors, stored at `data/random_vectors.h5`

## Preparing labelled data

The following steps are needed to prepared a labelled corpus for use in the system:


### Convert labelled data to format
Your labelled corpora must follow the directory structure of the provided example:
```
web ------------------------> top-level container
├── de --------------------->  class 1
│   ├── apollo8.txt -------->  document 1
│   └── rostock.txt
└── en --------------------->  class 2
    ├── needham.txt -------->  document 1
    └── thespis.txt
```	


### Process with Stanford pipeline
Process each labelled corpus with Stanford CoreNLP (you will likely have to: to increase the maximum amount of memory available to the JVM):

```
cd DiscoUtils
cd ~/projects/DiscoUtils
python discoutils/stanford_utils.py --data ../dc_evaluation/data/web --stanford ~/Downloads/stanford-corenlp-full-2015-04-20
```

You can control what processing is done by CoreNLP. Find the line that says `tokenize,ssplit,pos,lemma,parse,ner` and add/delete as required. Note NER is optional and very slow.

For our example data set, this produces a `web-tagged` directory, whose structure matches that of `web`. Files are CoNLL-formatted, e.g.:

```
1	He	he	PRP	O	2	nsubj
2	captained	captain	VBD	O	0	ROOT
3	the	the	DT	O	5	det
4	Australian	australian	JJ	MISC	5	amod
5	team	team	NN	O	2	dobj
6	in	in	IN	O	8	case
7	ten	ten	CD	NUMBER	8	nummod
8	Tests	test	NNS	O	2	nmod
9	,	,	,	O	_	_
10	winning	win	VBG	O	2	dep
11	five	five	CD	NUMBER	10	dobj
12	and	and	CC	O	10	cc
13	losing	lose	VBG	O	10	conj
14	five	five	CD	NUMBER	13	dobj
15	.	.	.	O	_	_
```

The columns are:

 - index
 - word
 - lemma
 - fine-grained PoS tag
 - NER tag
 -  index of dependency head
 - type of dependency relation

If an annotator (e.g. NER) is disabled, its corresponding column will contain only underscores.

### Convert to gzipped json (for faster access)
This extracts all features (words and phrases) of interest. You can select what you want to be extracted. Currently the only supported types are unigrams (filtered by PoS tag) and phrases (adjective-noun, noun-noun, verb-object or subject-verb-object compounds.).  Add your own phrase extractors as more models of composition become available. Edit `get_all_corpora` in `eval/data_utils.py` as required, then run:

```
cd ~/projects/dc_evaluation
python eval/scripts/compress_labelled_data.py --conf conf/exp0/exp0.conf --all --write-features
```
A configuration file containing tokenisation settings is required. This is detailed below. The output is (a compressed version of) the following:

```
["de", ["war/N", "zu/N"]]
["en", ["collaboration/N", "thespis/N", "extraher/N", "troupe/N"]]
```

One document per line, the first item in the list being the label, and the second item is a list of document features.

`compress_labelled_data.py` takes an additional boolean flag, `--write-features`, which is disabled by default. If true, a set of additional files will be written to `./features_in_labelled` for each labelled corpus. These contain a list of all extracted document features, a list of all noun phrase modifiers, a list of all verbs that appear in verb phrases, etc. I found these convenient during my PhD, as they make it easy to invoke compositional algorithms in batch. You may have no use for these files.

## Building word and phrase vectors
Use your preferred distributional method to build vectors for unigrams and phrases contained in all labelled corpora. These were extracted in the previous step.

During my PhD I used [this code](https://github.com/mbatchkarov/vector_builder) to build word and phrase vectors. See examples in that repository.

## Evaluating composed vectors

For the purposes of this example suppose we have processed a labelled classification corpus as described above and stored it to `data/web-tagged.gz`. We have also generated a random vectors for each document feature in the labelled corpus, and that they are stored in `data/random_vectors.h5`. We need a to write a configuration file to control the evaluation process. An example file is provided at `data/exp0/exp0.conf`. The file `conf/confrc` specifies the format of the configuration files and describes the meaning of each parameter. Configuration files are checked against the specification in `confrc` at the start of each experiments. You are ready to run an evaluation:

```
python eval/evaluate.py conf/exp0/exp0.conf
```

Results will appear in `conf/exp0.output`:

 - **exp0.conf**: copy of configuration file used to produce this output
 - **log.txt**: detailed experiment log
 - **exp0.scores.csv**: scores for each classifier for each cross-validation fold
 - **gold-cv0.csv**: gold-standard output for testing section of the labelled corpus
 - **predictions-*-cv0.csv**: predictions of each classifier for testing section of the labelled corpus

### Common types of experiments

- Non-distributional baseline
 
	 - decode_token_handler = eval.pipeline.bov_feature_handlers.BaseFeatureHandler
	 - must_be_in_thesaurus = False

- Standard feature expansion
 
	 - decode_token_handler = eval.pipeline.bov_feature_handlers.SignifierSignifiedFeatureHandler
	 - must_be_in_thesaurus = False

- Extreme feature expansion (EFE)
 
	 - decode_token_handler = eval.pipeline.bov_feature_handlers.SignifiedOnlyFeatureHandler
	 - must_be_in_thesaurus = False
	 - neighbours_file = data/random_vectors.h5 (something)

- Non-compositional EFE

	 - decode_token_handler = eval.pipeline.bov_feature_handlers.SignifiedOnlyFeatureHandler
	 - must_be_in_thesaurus = False
	 - neighbours_file = data/random_vectors.h5 (something)
	 - feature_extraction > train_time_opts > extract_unigram_features = J, N, V
	 - feature_extraction > train_time_opts > extract_phrase_features = ,
	 - feature_extraction > decode_time_opts > extract_unigram_features = J, N, V
	 - feature_extraction > decode_time_opts > extract_phrase_features = ,

### Common configuration pitfalls:
 
 - features extracted and train time do not overlap with (or are not distributionally comparable to) those at test time, e.g. nouns only at train time and verbs only at test time
 - feature selection too aggressive. This can be because `min_test_features` is too high, or because the distributional model (`neighbours_file`) does not contain vector for most of the document features. 
 - mismatch between preprocessing of labelled and unlabelled data, e.g. distributional vectors say `cat/NNS` and labelled documents say `cat/N`. Settings to watch are `use_pos`, `coarse_pos` and `lowercase`.

# Code

 Run unit tests with

 ```
 cd ~/projects/dc_evaluation
 py.test tests # OR python setup.py test OR runtests.py
 ```

# TODO
 - move tests to sane packages and reorganise 
 - separate phrase extraction from a dep tree to a new module
 - check all entry points take only cmd line args- no need to tweak source! 