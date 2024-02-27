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
* Scikit-Learn imputer function is used to impute missing values as follows:

        arr = np.array([[5,3,2,2],[3,None,1,9],[5,2,7,None]])
        imputer = SimpleImputer(strategy='mean')
        imp = imputer.fit(arr)
        imputer.transform(arr)
* A histogram is often a good visualization technique to use in order to see the overall behavior of a particular feature. You can create a histogram by using the Pandas 'hist' function, or Seaborn's 'distplot' function.
* The density plot plots the distribution of your single feature. The density plot is similar to a histogram but plots a smooth version of the histogram density using a kernel density function.
*  A high correlation between two attributes can sometimes lead to poor model performance.
* Scatterplot is a really good way to spot any special relationships among variables.
* Scatterplot matrices help you look at the relationship between multiple different features.
* Correlation matrices measure the linear dependence between features; they can be visualized with heat maps. For correlation, it can go as high as one, or as low as minus one. When the correlation is one, this means those two numerical features are perfectly correlated with each other. It's like saying Y is proportional to X. When those two variables’ correlation is minus one, it’s like saying that Y is proportional to minus X. Any linear relationship in between can be quantified by the correlation. So if the correlation is zero, this means there's no linear relationship—but it does not mean that there's no relationship. We can use Seaborn's heat map function to show the correlation matrix.
* Covariance is a measure of the relationship between two random variables and to what extent, they change together. The Pearson Correlation is just a covariance divided by the standard deviation of X and Y.

## Model Training
* The two most common formats supported by the Amazon SageMaker built-in algorithms are CSV and recordIO-protobuf.
* For Amazon SageMaker built-in algorithms, the label in your training dataset must be the first column on the left and your features should be to the right. Additionally, SageMaker requires that the CSV file have no header. However, some algorithms may not be able to work with training data in a data frame format.
* Amazon SageMaker automatically performs some transformations on your data. If you’re using Python, we recommend you use those transformations; if you’re using a different language, we recommend you use protobuf definition file that we provide in the AWS documentation.

**Testing and validation techniques:**
1. Simple hold-out: when you split your data into multiple sets, usually sets for training data, validation data, and testing data. Training data, which includes both features and labels.
2. Cross-validation: use cross-validation methods to compare the performance of multiple models. The goal behind cross-validation is to help you choose the model that will eventually perform the best in production:
   1. K-Fold cross-validation: for a small dataset, randomly partition the data into K different segments. For each segment, we’ll use the rest of the data outside of it for training in order to do a validation on that particular segment.
   2. Iterated K-Fold validation with shuffling
   3.  Leave-one-out cross-validation: the K is equal to N. Every time we leave one data point out for testing, we are using the rest in the training data. This is usually used for very small datasets where every data point is very valuable.
   4.  Stratified K-Fold cross-validation: ensure that for each fold, there are some equal weight proportions of the data for every different fold.
  
* You can use Sklearn to automatically split and shuffle the data at the same time.
* The loss function, which is sometimes called the objective function, is the measure of error in your model’s predictions given a set of weights
* Root mean square error (RMSE): Describes the sample standard deviation of the differences between predicted and observed values.
* Log likelihood loss (cross-entropy loss).
* Often, machine learning models are known to fall into what is called local minima, which prevents the model from improving more and thus preventing them from reaching the global minima.
* one of the simplest optimization techniques known as gradient descent. In this approach, the loss function is calculated at that step using the complete dataset, and the slope (also known as gradient) of the error curve is calculated using the loss function value at the current point.
* The step size, also known as “Learning rate” is a hyperparameter to the model. , a small value of gradient indicates the learning rate could be larger, which means we could safely take a larger step down and not go over the minimum. On the other hand, a large value of gradient indicates the slope is steep and, therefore, we must tread carefully and take small steps, so as not to fall off the cliff.

* Drawbacks to gradient descent:
  * Updates the parameters only after a pass through all the data (an epoch)
  * Can’t be used when the dataset is too large to fit entirely into memory
  * Can get stuck at local minima or fail to reach global minima

* In stochastic gradient descent, or SGD, you update your weights for each record you have in your dataset. For example, if you have 1,000 records or data points in your dataset, SGD will update the parameters 1,000 times. With gradient descent, the parameters would be updated only once—in every epoch. SGD leads to more parameter updates and, therefore, the model will get closer to the minima more quickly and with less overall computational power than gradient descent. One drawback of SGD, however, is that it will oscillate in different directions unlike gradient descent, which will always point towards the minima.
* Mini-batch gradient descent. This approach uses a smaller dataset or a batch of records (also called batch size) to update your parameters. Mini-batch gradient descent updates more than gradient descent while having less erratic/noisy updates when compared to SGD.
* Hyperparameters, compared to parameters(weights, bais), are external to a model and can’t be estimated from the data. Hyperparameters are set by humans. Think about hyperparameters as the knobs used to tune the ML algorithm to improve its performance.
*  Estimator, the high-level interface for Amazon SageMaker for model training. Estimators make it easy for you to specify the hardware you want for your training job including the container for your model, the training instance count and type of instance to use.
*  The fit() API calls the Amazon SageMaker CreateTrainingJob API to start model training. The API uses the configuration you provided to create the estimator and the specified input training data to send the CreatingTrainingJob request to Amazon SageMaker.

## Model Evaluation
* If your model is overfitting the training data, it makes sense to take actions that reduce model flexibility. To reduce model flexibility, try the following:
  * Feature selection: consider using fewer feature combinations, decreasing n-grams size, and decreasing the number of numeric attribute bins.
  * Increase the amount of regularization used.
* Variance: How dispersed your predicted values are
* Bias: The gap between predicted value and actual value
* Underfitting is where you have low variance and high bias. Overfitting is high variance and low bias.
* For classification problems, a confusion matrix is often used to classify why and how the model gets something wrong.
   * Accuracy (score): TP + TN / TP+TN+FP+FN   (less effective when there are a lot of true negative cases in your dataset)
   * Precision is the proportion of positive predictions that are actually correct. Best when the cost of false positives is high.
   * Recall (Sensitivity): TP / TP + FN (Best when the cost of false negatives is high)
   * Specificity: TP / TN + FN
   * F1 score helps express precision and recall with a single value: (2*precision*Recall)/(Precision+Recall)
   * AUC - ROC Curve: A performance measurement for a classification problem at various threshold settings. Uses sensitivity (true positive rate) and specificity (false positive rate)
     * AUC: Area-under-curve (degree or measure of separability).
     * ROC: Receiver-operator characteristic curve (probability curve)
* For Regression problems:
  * Mean Squared Error (MSE): take the difference between the prediction and actual value, square that difference, and then sum up all the squared differences for all the observations over N. In scikit-learn, you can use the mean squared error function directly from the metrics library.
  * R^2 = 1 - (Sum of Suared Error) / var(y) (commonly used metric with linear regression problems reporting a number from 0 to 1. When R squared is close to 1 it usually indicates that a lot of the variabilities in the data can be explained by the model itself. Always increasing when more variables are added to the model, which sometimes leads to overfitting. It isn’t always that the higher the R squared, the better the model. We have to balance the overfitting problem.)
  * Adjusted-2 is a better metric for multiple variates regression. The Adjusted R squared has already taken care of the added effect for additional variables and it only increases when the added variables have significant effects in the prediction.  R^2 will always increase when more explanatory variables are added to the model; the highest R^2 may not be the best model.

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






    
    
    

