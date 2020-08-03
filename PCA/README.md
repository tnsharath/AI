# PCA Algorithm:

A simple form of PCA algorithm with logistic regression.

### Steps:
1. Seperate X and Y from data set and also split data into X_train and X_test.
2. X(i) <- X(i) - mu(i)
3. If required scale the features (X) with Standard deviation. Call this as Z.
4. Multiply Z transpose and Z. compute the co varience of the result.
5. Find the eigenvalues and eigenvectors of covarience matrix.
6. Sort the eigenvalues in descending order based on eigenvalues sort respective eigenvectors. Call sorted eigenvectors as P_star.
7. multiply Z and P_star. which will give the Principle components.
8. Decide on number of Principle Components to consider.
