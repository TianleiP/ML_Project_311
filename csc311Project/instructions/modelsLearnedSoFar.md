K Nearest neighbours
Decision Tree
Linear Regression
Logistic regression
Neural network
Naive Bayes
Gaussian Discriminant analysis
Clustering KMeans
Gaussian Mixture model

Not Suitable Models:
Linear Regression - Designed for continuous outputs, not for classification tasks like ours that require categorical outputs.

Clustering (KMeans) - This is an unsupervised learning technique for grouping similar data points. Our task is supervised classification with known labels.

Gaussian Mixture Model - Also primarily an unsupervised learning method for density estimation and clustering, not suited for direct classification.

Neural Network - Can work well for this task, But super hard to implement and debug without using sklearn

Suitable Models:
K-Nearest Neighbors - Excellent for this classification task, especially with our numerical and one-hot encoded features.

Decision Tree - Very suitable for classification and can handle mixed data types well. Easy to implement without sklearn.

Logistic Regression - Perfect for multi-class classification and can be implemented relatively easily with numpy.

Naive Bayes - Great for classification tasks and relatively simple to implement without sklearn.

Gaussian Discriminant Analysis - Can work well for this classification task and can be implemented with numpy.