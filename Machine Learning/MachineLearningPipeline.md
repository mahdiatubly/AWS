## Feature Engineering and Model Tuning
**Feature engineering:** is the science (and art) of extracting more information from existing data in order to improve your model’s prediction power and help your model learn faster.

**Dimensionality:** means the number of features (or inputs) you have in your data set. The phrase curse of dimensionality refers to the fact that models will have a difficult time finding the patterns you want them to identify when there are many different dimensions of data (many features) to sort through.

* Think of feature engineering as being made up of three similar, yet slightly different,
processes:
    - Feature extraction:
      - In natural language processing: it could be extracting useful features like the most popular words from text that aren’t articles or prepositions.
      - With Structured data: Principal component analysis (PCA) or t-distributed stochastic neighbor embedding (T-SNE)

    - Feature selection:  Filtering the data is one of the common techniques that you will use for feature selection. Remember, machine learning algorithms are not only used for typical structured datasets. Oftentimes, we're dealing with images or audio, for instance. For those types of data formats, the data structure is more complicated and therefore often requires filtering to be more specific to our business problem.
    - Feature creation and transformation

