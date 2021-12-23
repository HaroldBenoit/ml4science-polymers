# Documentation

## Overview
This repo contains the work done by (*Lucas Trognon, Harold Benoit, Mahammad Ismayilzada*) to use machine learning to quantify and compare the difficulty of identifying different polymer sequences, as described above. Using this information, we will assess some pitfalls with different encodings (different backbone: ‘0’ ~ ‘6’ or double-encoding: ‘2’ ~ ‘22’) and ultimately find a basis to encode information in these polymers (bulky elements ‘1’ ~ ‘2’ ~ ‘5’). 

For a quick introduction on the concept of polymers in the context of digital storage, you may read the introduction of our report or [this article](https://actu.epfl.ch/news/bacterial-nanopores-open-the-future-of-data-stor-6/) for a lighter read.

This project was done in the context of the ML4Science project, where students can bring their ML skills to practice by joining forces with any lab on the EPFL campus, or other academic institutions, across any disciplines and topics of their choice. 

This project was done in collaboration with the EPFL Laboratory for Biomolecular Modeling.


## Repo Structure
This repo is organized as following:
* `requirements.txt` - This file contains all the dependencies to run this project.
* `pipeline.py` - This file contains the implementation of our data processing pipeline to go from polymer reading events of varied lengths to data usable by ML algorithms, especially neural networks. For a comprehensive overview of the inner workings of the pipeline, we refer the reader back to the *Data processing pipeline* section of our report.
* `models.py` - This file contains the implementation of the neural networks model used in our work. Additionally, helper functions specific to neural networks can be found there.
* `helpers.py` - This file contains the code for various helper functions to complement ML algorithms and perform exploratory analysis, preprocessing, feature engineering.
* `ml4science-report.pdf` - 4 page report summarizing the work done in this repo. 
* **`multi-class`** - This folder contains the research done on distinguishing between the 3 bulky elements: '2', '4' and '5'.
* **`double-encoding`** - This folder contains the research done on distinguishing between using a single bulky element '2' and a double bulky element '22'.
* **`backbone`** - This folder contains the research done on distinguishing between using the '0' backbone and the '6' backbone.


## Setup
This project requires python >= 3.7 and dependencies can be installed using the `requirements.txt` in your favorite python virtual environment:
```sh
pip install -r requirements.txt
```

## Project

### Outline

For each research question, the structure is as follows:

* Exploratory data analysis
* Testing different models
* Hyperparameter tuning for the best model


### Exploratory Data Analysis

* The lengths of the relative current time-series span a wide range of values, ranging from 0.3 milliseconds to  2 seconds. One reason is that some polymers get stuck in the nanopore and do some sort of back and forth in it before finally leaving.
* Two events are indistinguishable for the human eye. Both in time series, autocorrelation, and frequency domain representations.
* The relative current sensor has a slow response time in its measurements. Thus the relative current measured is highly dependent on the speed of the polymer passing through. 

### Preprocessing and Feature Engineering


### Model Selection and Hyperparameter Tuning

