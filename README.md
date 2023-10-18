# Perspectives: Directed Emotion Extraction

## Overview

The Perspectives library offers an easy way to extract directed emotions at scale with its [cutting edge emotion extraction model](https://huggingface.co/helliun/bart-perspectives). It is built on top of the powerful pandas DataFrame functionality. The library introduces several novel methods for text analytics, perfect for dealing with customer feedback, analyzing semantic trends, or profiling entities within a text.
![image](https://github.com/helliun/perspectives/blob/main/burr_perspective.png)

## Main Features

1. **Easily extract perspectives from text**: The `get_perspectives()` function allows you to extract the speaker's identity, emotions, and the object of these emotions, giving you useful insights about the emotions in your text.

2. **Powerful search capabilities**: You can search the dataset based on the speaker, emotion, object, and reason. The search method leverages the sentence transformer models for semantic search functionality, providing you with outputs that are spot-on.

3. **Improved machine learning models**: The library efficiently interfaces with PyTorch BART model for Seq2Seq learning and the mpnet-base model for sentence-transformations, which provides powerful text representation for ML models.

4. **Structured emotional outcomes**: All outputs are neatly structured in DataFrame format, allowing for easy downstream analysis and visualizations.

## Installation

	pip install perspectives

## Usage

```python
from perspectives import DataFrame

# Load DataFrame
df = DataFrame(texts = [list of sentences]) 

# Load model
df.load_model() 

# Get perspectives
df.get_perspectives()

# Search
df.search(speaker='...', emotion='...')
```

[Full colab demo](https://colab.research.google.com/drive/1asovKRUHmsZfZo8Iz18q_dfAJXzahhmB?usp=sharing)

## Contributing and Support

Contributions are welcome! Please raise a GitHub issue for any problems you encounter.

## Licence

The library is open source, free to use under the MIT license. 

Please note that this library is still under active development, hence you may see regular updates and improvements. Feel free to contribute!
