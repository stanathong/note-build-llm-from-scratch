# Note taken while reading "Build A Large Language Model from Scratch"

This repo contains my study notes based exclusively from the book [Build A Large Language Model from Scratch by Sebastian Raschka](https://amzn.to/4fqvn0D). The majority of images are taken from the book itself.

## Notes and Extra Contents for Each Chapter

| Chapter | Link | Status |
| --- | --- | ---|
| **Chapter 1** Understanding large language models | [ch1/README.md](ch1/README.md) | **Completed** |
| **Chapter 2** Working with text data | [ch2/README.md](ch2/README.md) | **Completed** |
| **Chapter 3** Coding attention mechanisms | [ch3/README.md](ch3/README.md) | **Completed** |
| **Chapter 4** Implementing a GPT model from scratch to generate text | [ch4/README.md](ch4/README.md) | **Completed** |
| **Chapter 5** Pretraining on unlabeled data | [ch5/README.md](ch5/README.md) | **In progress** |
| **Chapter 6** Fine-tuning for classification | [ch6/README.md] | **Not started** |
| **Chapter 7** Fine-tuning to follow instructions | [ch7/README.md] | **Not started** |
| **Workshop** Building LLMs from the Ground Up: A 3-hour Coding Workshop | [workshop/README.md](workshop/README.md) | **Completed** |

## Pre-requisite

1. Clone the repository

```
git clone --depth 1 https://github.com/rasbt/LLMs-from-scratch.git
```

2. Set up the environment for developing an llm, named llm-book-311

```
conda create --name llm-book-311 -y python=3.11
conda activate llm-book-311

pip install -r requirements.txt
```

For reference, requirements.txt is defined as

```
torch >= 2.0.1        # all
jupyterlab >= 4.0     # all
tiktoken >= 0.5.1     # ch02; ch04; ch05
matplotlib >= 3.7.1   # ch04; ch05
tensorflow >= 2.15.0  # ch05
tqdm >= 4.66.1        # ch05; ch07
numpy >= 1.25, < 2.0  # dependency of several other libraries like torch and pandas
pandas >= 2.2.1       # ch06
psutil >= 5.9.5       # ch07; already installed automatically as dependency of torch
```

In this setting, we will no longer use Jupyter Notebook but Jupyter Lab. Execute by running on the command line as `jupyter-lab`.

3. Install [MarkText](https://www.marktext.cc/) to be used a Markdown editor
4. To setup docker, follow https://github.com/rasbt/LLMs-from-scratch/tree/main/setup/03_optional-docker-environment






