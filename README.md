# NLP coursework

Two notebook-based projects from an NLP course at National Tsing Hua University.

## Word embedding analogies

Compared pretrained GloVe with Word2Vec trained on three nested samples of English Wikipedia and on the Brown corpus. The evaluation covers 19,544 Google analogy questions and reports accuracy and vocabulary coverage by category.

| Model | Overall accuracy | Coverage |
| --- | ---: | ---: |
| GloVe | 63.11% | 100.00% |
| Word2Vec, Wikipedia 5% | 52.92% | 98.79% |
| Word2Vec, Wikipedia 10% | 55.54% | 99.08% |
| Word2Vec, Wikipedia 20% | 56.70% | 99.08% |
| Word2Vec, Brown | 0.87% | 32.26% |

Accuracy counts uncovered questions as incorrect. See the [notebook](word-embeddings/experiment.ipynb), [category results](word-embeddings/comparison.csv), and [comparison chart](word-embeddings/model_comparison.png).

## Character-level LSTM for arithmetic

Trained a PyTorch LSTM to generate answers one character at a time from expressions involving addition, subtraction, multiplication, and parentheses. The notebook uses teacher forcing during training and greedy decoding for evaluation. The recorded run reached **86.33% exact match** over 10,000 evaluation examples after 10 epochs.

See the [notebook](lstm-arithmetic/experiment.ipynb) for model code, training, and error analysis.

## Running the notebooks

Open each notebook from its own directory in a Python 3.13 Jupyter environment. Each project has a requirements file recording the original environment. The arithmetic run used a CUDA build of PyTorch. Use the PyTorch build appropriate for your platform.

- The embedding notebook downloads the analogy benchmark, pretrained vectors, and course-provided Wikipedia shards. The Wikipedia experiment uses substantial disk space and depends on the original download links.
- The arithmetic notebook expects course-provided training and evaluation files at `data/arithmetic_train.csv` and `data/arithmetic_eval.csv`, with columns `src` and `tgt`. Set `resume_training = False` to train from scratch and set the epoch count to 10 to match the recorded run.

The repository contains the experiment code and recorded results. Datasets, model weights, and submission archives are stored separately.
