TextGAN
=======
Implementation of [TextGAN](https://c4209155-a-62cb3a1a-s-sites.googlegroups.com/site/nips2016adversarial/WAT16_paper_20.pdf?attachauth=ANoY7cqpAyY5CnhFXJnMkCb5JpTtM-SAdM3a4lGtDwHTc9Zgk1_S4ARZEA-GChW9mUOEN13e58IlNJHZER3DxCvDrRJSayUeM-Ss9rAxYl7eTVCUtzyxoI53o2lBASgxjnGammqZB8XODyoMwO_mjKSgTA2eMAih2nXVG9XyEugbJ2FfoEj4YEw-RxOPVOzzY55zvyHBA6DmnNRnlFn6e7s_pgUu5vySPGse-6EUi4aWkI-kFo5pl9E%3D&attredirects=0) using Tensorflow

Requirements
------------
- Python 2.7+, 3.4+
- Tensorflow 1.1
- NLTK
- spaCy
- parlAI
- SimpleQuestions
- [Gutenberg](https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html)
- [GloVe](https://nlp.stanford.edu/projects/glove/)

Troubleshooting
---------------
- `ValueError: unknown local: UTF-8`

```bash
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
```

- `ValueError: Word vectors set to length 0. This may be because the data is not installed. If you haven't already, run python -m spacy download en to install the data.`

```bash
python -m spacy download en
```

- `Resource u'tokenizers/punkt/english.pickle' not found.  Please use the NLTK Downloader to obtain the resource:`

```bash
python3 -m nltk.downloader punkt
```
