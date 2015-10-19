# Evaluation via document classification


## Preparing labelled data

The following steps are needed to prepared a labelled corpus for use in the system:



### Convert labelled data to mallet format
Directory structure of the labelled data
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
Process with Stanford CoreNLP (you will likely have to: to increase the maximum amount of memory available to the JVM):

```
cd DiscoUtils
# edit stanford_utils.py and specify where your data and copy of CoreNLP are. See comments therein.
python discoutils/stanford_utils.py
```

You can control what processing is done by CoreNLP. Find the line that says `tokenize,ssplit,pos,lemma,parse,ner` and add/delete as required. Note NER is very slow.

For our example data set, this produces a `web-tagged` directory, whose structure matches that of `web`. Files are CoNLL-formatted:

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
 - index of dependency head
 - type of dependency relation

If an annotator (e.g. NER) is disabled, its corresponding column will contain only underscores.

### Convert to gzipped json (for faster access)
This extracts all features (words and phrases) of interest. You can select what you want to be extracted. Currently the only supported types are unigrams (filtered by PoS tag) and phrases (adjective-noun, noun-noun, verb-object or subject-verb-object compounds.)

Edit `get_all_corpora` in `eval/data_utils.py` as required, then run:

```bash
python eval/scripts/compress_labelled_data.py --conf conf/exp0/exp0.conf --all
```

PyPy might speed this part significantly. The output is (a compressed version of) the following:

```
["de", ["war/N", "zu/N"]]
["en", ["collaboration/N", "thespis/N", "extraher/N", "troupe/N"]]
```

One document per line, the first item in the list being the label, and the second item is a list of document features.

### Extract phrases to compose

## Building word and phrase vectors

## Evaluating composed vectors