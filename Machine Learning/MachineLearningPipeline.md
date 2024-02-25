## Sagemaker
A fully managed service used to build, train, and deploy ML models at any scale.
* Amazon SageMaker Ground Truth provides access to teams of human labelers to label your data, then you can feed that human-labeled data back to the service for automatic labeling – where it actually uses ML to learn to label the data in the same way that the humans did.
* Feature is an attribute that can be used to help identify patterns and predict future answers.
* Label is the answer that you want your model to predict.
* Model performance metrics typically expressed in terms of accuracy.
* If your training data is already in the Amazon Elastic File System (Amazon EFS), we recommend using that as your training data source. Amazon EFS has the benefit of directly launching your training jobs from the service without the need for data movement, resulting in faster training start times.

## Data Processing
* Pandas, an open-source Python library, can be used for this particular data reformatting. Pandas reformat data from various formats like CSV, JSON, Excel, Pickle, and others into a tabular representation, presenting it in rows and columns.
* A data frame is made up of many series, which are essentially columns capable of holding any data type. The axis labels are referred to as the index.
* you can call the shape object to see your dataset’s dimensions. Shape returns the number of rows and columns. Columns return the names and types of your columns, and Rows do the same for rows.
* NumPy For objects with multi-dimensional arrays
* Scikit-learn For data mining and data analysis with Python Built on NumPy, SciPy, and Matplotilb libraries
* Matplotlib A visualization library for Python used for two-dimensional plots of NumPy arrays
* Seaborn is another visualization library for Python. Built on top of Matplotlib and Closely integrated with Pandas DataFrames
* df.describe() Generates descriptive summaries of your numerical data, such as count, mean, std, min, max.
* df.describe(include='all') Generates descriptive summaries of your numerical and categorical data.
* Transforming the outlier: You could do this by taking the natural log of a value, which in turn would reduce the variation caused by the extreme outlier value and therefore the outlier’s influence on the overall dataset.
* Many learning algorithms can’t handle missing values.
* Check how many missing values for each column: df1.isnull().sum()
* Default drops the rows with NULL values df1.dropna(). “axis=1” drops the columns with NULL values df1.dropna(axis=1)









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
3. Data Hyperprameters: Define attributes of the data itself. define data augmentation techniques like cropping or resizing for image-related problems. Useful for small/homogenous datasets.

**Methods for tuning hyperparameters:**
1. Grid search: you set up a grid made up of hyperparameters and their different values. For each possible combination, a model is trained and a score is produced on the validation data. With this approach, every single combination of the given possible hyperparameter values is tried. This approach, while thorough, can be very inefficient.
2. Random search: similar to grid search, but instead of training and scoring on each possible hyperparameter combination, random combinations are selected. You can set the number of search iterations based on time and resource constraints.
3. Bayesian search: Makes guesses about best hyperparameter combinations, then uses regressions to refine the combinations.


* Amazon SageMaker lets you perform automated hyperparameter tuning. Automatic model tuning can be used with the Amazon SageMaker built-in algorithms, pre-built deep learning frameworks, and bring-your-own-algorithm containers. During hyperparameter tuning, Amazon SageMaker attempts to figure out if your hyperparameters are log-scaled or linear-scaled. Initially, it assumes that hyperparameters are linear-scaled. If they should be log-scaled, it might take some time for Amazon SageMaker to discover that. If you know that a hyperparameter should be log-scaled and can convert it yourself, doing so could improve hyperparameter optimization.

* Running more hyperparameter tuning jobs concurrently gets more work done quickly, but a tuning job improves only through successive rounds of experiments. Typically, running one training job at a time achieves the best results with the least amount of computing time.



## Model Deployment

**To Deploy and host on Amazon SageMaker:** 
1. Create the model: Use the CreateModel API, Name the model, and tell Amazon SageMaker where it is stored.
2. Create an HTTPS endpoint configuration: Use the CreateEndpointConfig API. Associate it with one or more created models. Set one or more configurations (production variants) for each model, For each production variant, specify instance type and initial count, and set its initial weight (how much traffic it receives).
3.  Deploy an HTTPS endpoint based on an endpoint configuration: Use the CreateEndpoint API.

**Deployment Techniques:**
1. **Blue/green deployment technique:** provides two identical production environments. You can use this technique when you need to deploy a new version of the model to production. This technique requires two identical environments:
    * A live production environment (blue) that runs version n.
    * An exact copy of this environment (green) that runs version n+1.
2. **Canary deployment:** Compare the performance of different versions of the same feature while monitoring a high-level metric.
3. **A/B testing:** is similar to canary testing, but has larger user groups and a longer time scale, typically days or even weeks.

**Amazon SageMaker supports four inference option types:**
1. Batch transform: provides offline inference for large datasets. Helpful for preprocessing datasets or gaining inferences from large datasets. It also is useful when running inference if a persistent endpoint is not required, or associating input records with inference to support the interpretation of results.
2. Real-time inference: is ideal for inference workloads where you have real-time, interactive, low latency requirements. Fully managed and supports autoscaling.
3. Serverless inference: can be used to serve model inference requests in real time without directly provisioning compute instances or configuring scaling policies.
4. Asynchronous inference queues incoming requests for asynchronous processing. This option is ideal for requests with large payload sizes (up to 1 GB), long processing times (up to 1 hour), and near-real-time latency requirements.

* Amazon Elastic Inference enables you to attach low-cost, GPU-powered acceleration to Amazon EC2 and Amazon SageMaker instances to reduce the cost of running deep learning inferences.
* AWS IoT Greengrass enables you to perform ML inferencing locally on devices, using models that are created, trained, and optimized in the cloud.
* The Amazon SageMaker Neo compiler and a runtime help solve the need for optimized models for each hardware. The compiler converts models to an efficient, common format. The runtime is optimized for the underlying hardware and uses specific instruction sets that speed up ML inference.






    
    
    

