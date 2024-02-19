## Feature Engineering and Model Tuning
**Feature engineering:** is the science (and art) of extracting more information from existing data in order to improve your model’s prediction power and help your model learn faster.

**Dimensionality:** means the number of features (or inputs) you have in your data set. The phrase curse of dimensionality refers to the fact that models will have a difficult time finding the patterns you want them to identify when there are many different dimensions of data (many features) to sort through.

* Think of feature engineering as being made up of three similar, yet slightly different,
processes:
    - Feature extraction:
      - In natural language processing: it could be extracting useful features like the most popular words from text that aren’t articles or prepositions.
      - With Structured data: Principal component analysis (PCA) or t-distributed stochastic neighbor embedding (T-SNE)

    - Feature selection:  Filtering the data is one of the common techniques that you will use for feature selection. Remember, machine learning algorithms are not only used for typical structured datasets. Oftentimes, we're dealing with images or audio, for instance. For those types of data formats, the data structure is more complicated and therefore often requires filtering to be more specific to our business problem.
    - Feature creation and transformation: is the process of generating new features from existing features.
         - For numerical features: the techniques include:
           - Taking the log, square root, or cube root of the feature.
           - Binning is a great strategy that puts continuous data into groups, also called bins. This is one way to convert a continuous variable into a categorical variable. Binning has a smoothing effect on the input data and may also reduce the chances of overfitting in cases with small datasets.
           - Scaling is a technique that is applied to each feature. Scaling is important because a lot, although not all, machine learning algorithms are very sensitive to different ranges of data. Sometimes, a wide range of data will lead to optimization failure. There are exceptions, however. Decision trees in the random forest algorithm, are usually not sensitive to the scale of the variables in the dataset. In general, we're applying a transformation to a particular column, and different columns are scaled independently. We are only using the data in a specific column to do the scaling.
Scaling transformation techniques:
                - Mean-variance standardization: For one particular feature X, we are going to first find out the mean and standard deviation for that particular feature in the training dataset. What we're going to do is remove the mean and divide by the standard deviation for each observation. After scaling, the new feature is going to have a mean of zero, and a standard deviation of one. To achieve this transformation easily, we can call the scikit-learn StandardScaler function.
by using the mean-variance standard deviation procedure, we still keep the outliers, but reduce the impact of those outliers dramatically.

         - 

