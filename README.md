# SpaceY Falcon 9 Project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

>This project was completed as the capstone for IBM's Data Science Professional Certificate. The business scenario 
(competing launch provider "SpaceY") is a fictional framing provided by the course; all underlying data and analysis 
are based on real, publicly available SpaceX launch information.

## Predicting Falcon 9 First Stage Reuse -- A SpaceY Case Study
The commercial space industry has shifted dramatically over the past decade, with private companies like Virgin Galactic,
Rocket Lab, and Blue Origin now competing alongside government agencies. SpaceX has emerged as the clear leader, 
and much of that edge comes down to cost: a Falcon 9 launch is advertised at \$62 million, compared to \$165 million 
or more from competing providers. The difference is largely explained by one thing: SpaceX recovers and reuses the 
rocket's first stage instead of discarding it after every flight.

This project takes the perspective of a data scientist at SpaceY, a fictional new launch provider looking to compete 
with SpaceX. Since the cost of a launch depends heavily on whether the first stage can be recovered, the core question 
becomes: **can we predict, using only publicly available data, whether a Falcon 9 first stage will be successfully reused?** 
Understanding this is the foundation for estimating a competitive launch price.

To answer this, launch data was combined from multiple public sources (SpaceX's launch history via the Launch Library 
2 API, the General Catalog of Artificial Space Objects, and supplementary mission data) into a single reproducible 
pipeline. From there, the project moves through exploratory analysis (statistical and geospatial, including each 
launch site's proximity to coastlines, highways, and railways), an interactive dashboard, and finally a set of 
classification models trained to predict landing outcomes. A Support Vector Classifier ultimately performed best, 
correctly identifying every successful landing in the test set with only a small number of false positives — with 
payload mass, orbit type, and booster version emerging as stronger predictors than launch site location alone.

## Project Organization

```
├── LICENSE            <- Open-source license
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         spacey_falcon_9_project and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── spacey_falcon_9_project   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes spacey_falcon_9_project a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

