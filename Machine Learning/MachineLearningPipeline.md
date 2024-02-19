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
                - Mean-variance standardization: For one particular feature X, we are going to first find out the mean and standard deviation for that particular feature in the training dataset. What we're going to do is remove the mean and divide by the standard deviation for each observation. After scaling, the new feature is going to have a mean of zero and a standard deviation of one. To achieve this transformation easily, we can call the scikit-learn StandardScaler function. By using the mean-variance standard deviation procedure, we still keep the outliers but reduce the impact of those outliers dramatically.
                - MinMax scaling: we subtract the minimum value of the feature and divide it by the difference between the maximum of the feature and the minimum of the feature. After the transformation, the new variable is going to be between zero and one. MinMax scaling can be easily done by calling the MinMax scaler from the scikit-learn package. This transformation does not change the distribution of the feature and due to the decreased standard deviation, the effects of the outliers increase. Therefore, before transformation, it is recommended to handle the outliers. Robust to small standard deviations, When the standard deviation is small the mean-variance scaling is not going to be robust because we're dividing by a very small number.
                - MaxAbs scaling: Divides all the data in a feature by the maximum absolute value found in that feature. doesn't destroy sparsity because we don’t center the observation through any measurement.
                - Robust scaling: Subtracts the median of the feature and divides that by the difference between the 75th and 25th quartile. Minimizes impact of large marginal outliers.
      


         - For Categorical features:
           - Ordinal: For ordinal variables like this one, you can use a map function in Pandas to convert the text into numerical values.
           - Nominal: With one-hot encoding, you convert this one column of Home Types into three columns: A column for House, a column for Apartment, and a column for Condo. Pandas get_dummies function will automatically create the new columns based on one-hot encoding and create the column names for you, with the entry for each category at the end of the variable name.


### Model Tuning

**Hyperparameter categories:**
    1. Model Hyperparameters: Help define the model. Such as Filter size, pooling, stride, and padding.
    2. Optimizer Hyperparameters: How the model learns patterns on data. Such as Gradient descent, and stochastic gradient descent.
    3. Data Hyperprameters: Define attributes of the data itself. Useful for small/homogenous datasets.
**Methods for tuning hyperparameters:**
    1. Grid search: you set up a grid made up of hyperparameters and their different values. For each possible combination, a model is trained and a score is produced on the validation data. With this approach, every single combination of the given possible hyperparameter values is tried. This approach, while thorough, can be very inefficient.
    2. Random search: similar to grid search, but instead of training and scoring on each possible hyperparameter combination, random combinations are selected. You can set the number of search iterations based on time and resource constraints.
    3. Bayesian search: Makes guesses about best hyperparameter combinations, then uses regressions to refine the combinations.

* Amazon SageMaker lets you perform automated hyperparameter tuning. Automatic model tuning can be used with the Amazon SageMaker built-in algorithms, pre-built deep learning frameworks, and bring-your-own-algorithm containers.

    
    
    

