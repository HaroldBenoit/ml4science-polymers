
Compilation of thoughts and notes on the project

## Research questions:

1. How to select / filter events? Is it necessary?
2. Is it better to extract hand-picked features (+ NN, boosting, etc.) or use the raw signal and let the model extract patterns (CNN, RNN, or transformers)?
3. Can you distinguish between '2', '4', '5' for the same sequence?
4. Can you distinguish between '020' and '0220'? => sensitivity of the method
5. Can you distinguish between encoding the same polymer with '0' or '6'? => the difference in the molecule is very small it would indicate a high sensitivity of the method.
(I didn't send the ones with only '0' and only '6' yet but we should have the data) Which encoding is the easiest to read/classify?

## Data details

How to deal with varied length inputs when training and classifying ?

Givent that the open current (100% relative current) is the default state, it seems reasonable to use padding with this default value. The question is do we do pre-sequence padding or post-sequence padding ? Some sequences are really long and we are fairly sure that there's enough information such that they can be truncated a little. (https://machinelearningmastery.com/data-preparation-variable-length-input-sequences-sequence-prediction/)

A reasonable approach would be to pad the shortest and truncate the longest to arrive at some mean length that we can use in our vectorized algorithms.

It is also possible after the padding preprocessing to add a masking layer to our network to explicitly ignore the padded values (https://machinelearningmastery.com/data-preparation-variable-length-input-sequences-sequence-prediction/) using Keras at least.

## Methods

Entire tutorial on machine learning with time series (https://www.youtube.com/watch?v=wqQKFu41FIw)

### Feature extraction

1. Component analysis
2. Wavelet analysis
3. High-pass and low-pass filter
4. Clustering


### ML methods

- Deep learning 
    - CNN
    - RNN (variable length input)
        - LSTM !! especially suited for time series data
    - Transformers

- Random forest
- Boosted trees
- SVM / Logistic reg
