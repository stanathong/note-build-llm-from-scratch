# note-build-llm-from-scratch

## Pre-requisite

1. Clone the repository

```
git clone --depth 1 https://github.com/rasbt/LLMs-from-scratch.git
```

2. Set up the environment for developing an llm, named llm-book-311

```
conda create --name llm-book-311 -y python=3.11
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

3. Install [MarkText](https://www.marktext.cc/) to be used a Markdown editor
4. To setup docker, folwer https://github.com/rasbt/LLMs-from-scratch/tree/main/setup/03_optional-docker-environment


