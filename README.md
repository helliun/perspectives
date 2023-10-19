# Perspectives: Pandas-based library for emotion graphing and semantic search with LMs

## Overview

The Perspectives library offers an easy way to extract perspectives (emotion events with a speaker, emotion, object, and reason) at scale with its [cutting-edge emotion extraction model](https://huggingface.co/helliun/bart-perspectives). It is built on top of the powerful pandas DataFrame functionality, with added support for semantic search. The library introduces several novel methods for text analytics, **perfect for dealing with customer feedback, analyzing semantic trends, or profiling entities within a text.**
![image](https://github.com/helliun/perspectives/blob/main/burr_perspective.png)
*Graph generated from extracted perspectives*


![image](https://github.com/helliun/perspectives/blob/main/burr_search_example.png)
*Semantic search dashboard built on top of pandas*

## Main Features

1. **Easily extract perspectives from text**: The `get_perspectives()` function allows you to **extract the speaker's identity, emotions, and the object of these emotions**, giving you useful insights about the emotions in your text.

2. **Powerful search capabilities**: You can perform semantic search on the dataset based on **any column or combination of columns** in the dataset (including columns generated from perspective extraction) . The search method leverages the sentence transformer models for semantic search functionality, providing you with outputs that are spot-on.

3. **Improved machine learning models**: The library efficiently interfaces with specialized model `bart-perspectives` for extraction and the mpnet-base model for search.

4. **Structured emotional outcomes**: All outputs are neatly structured in DataFrame format, allowing for easy downstream analysis and visualizations.

## Installation

	pip install perspectives

## Usage

```python
from perspectives import DataFrame

# Load DataFrame
df = DataFrame(texts = [list of sentences]) 

# Get perspectives
df.get_perspectives()

# Semantic search on any combination of columns
df.search(speaker='...', emotion='...')

# Profile
df.graph(speaker='...')
```
### Demo

[Video demo](https://github.com/helliun/perspectives/assets/65739931/a9270e80-1b11-43d6-8330-e7589ef06438)



[Colab demo for profiling](https://colab.research.google.com/drive/1asovKRUHmsZfZo8Iz18q_dfAJXzahhmB?usp=sharing)

[Colab demo for analyzing customer reviews](https://colab.research.google.com/drive/1XNWUqJbDNSLJz5kRyeQZaJyLaS_U2BG-?usp=sharing)

## About me

I'm a recent grad of Ohio State University where I did an undergraduate thesis on Synthetic Data Augmentation using LLMs. I've worked as an NLP consultant for a couple awesome startups, and now I'm looking for a role with an inspiring company who is as interested in the untapped potential of LMs as I am! [Here's my LinkedIn.](https://www.linkedin.com/in/henry-leonardi-a63851165/)

## Contributing and Support

Contributions are welcome! Please raise a GitHub issue for any problems you encounter.

[Buy me a coffee!](https://www.buymeacoffee.com/helliun)

## Licence

The library is open source, free to use under the MIT license. 

Please note that this library is still under active development, hence you may see regular updates and improvements. Feel free to contribute!
