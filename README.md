# Credibility Classification of Credit Card Clients

#### Contributors

- Chester Wang
- HanChen Wang
- Qurat-ul-Ain Azim
- Renee Kwon

## About
In the field of risk management, one of the most common problem is default prediction. This allows companies to predict the credibility of each person, analysis the risk level and optimizing decisions for better business economics. In this project, we aim to learn and predict a person's credibility based on his/her basic personal information (gender, education, age etc. ).



## Research Question
Through this project, we aim to answer the question: Which attributes are most important when we use machine learning models to predict the default? Specifically we would love to know if the `weight` of attributes would change when we employ different models. Answering this question is, from our perspective, of great importance because it allows to understand what attributes relate to the credibility most.

### Data Summary
We use a dataset hosted by the UCI machine learning repository. Originally it is collected by researchers from Chung Hua University and Tamkang University. As the probability of default cannot be actually acquired, the targets are obtained through estimation as stated by the authors of this dataset. The dataset consists of 30000 instances, with each consists of 23 attributes and a target. The raw dataset is about 5.5 MB large, and we split it into the training set (80%) and testing set (20%) for further use.

### Data Analysis Plan
Building machine learning models and score the performance of each model would be our major method to approach the dataset. Specifically to study the influence of each attribute, we will look into the models based on their performance and order the importance of attributes with weights.

We plan to explore a wide range of algorithms (both naive and advanced), including:
- Support Vector Machine
- Logistic Regression
- Decision Tree
- Ensemble Methods (e.g. Random Forest)
- KNN (both unweighted and weighted)

Most of our code for data processing and model building and will be in Python 3. For EDA and data visualization we plan to use both R and Python 3.


### EDA Plan



## Usage

The automated data download can be performed via running the following command using terminal:

`python data_downloader.py "https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients" <PATH>`

Further details on the analysis will be updated as the analysis is performed. 


## License
We use MIT license for this project.


## References

- Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.
