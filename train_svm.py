import numpy as np
from numpy.linalg import inv, norm
from scipy.optimize import minimize
import pickle

class RBF_SVM:
    def __init__(self, C=1.0, gamma=0.5):
        self.C = C
        self.gamma = gamma

    def rbf_kernel(self, X1, X2):
        # Compute RBF kernel between two sets of vectors
        sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * sq_dists)

    def fit(self, X, y):
        print("Fitting")
        n_samples, n_features = X.shape
        K = self.rbf_kernel(X, X)
        
        # Construct the matrices for the quadratic programming problem
        P = np.outer(y, y) * K
        q = -np.ones(n_samples)
        G = np.vstack([np.eye(n_samples) * -1, np.eye(n_samples)])
        h = np.hstack([np.zeros(n_samples), np.ones(n_samples) * self.C])
        
        print("Finished construction")

        # Solve the quadratic programming problem using scipy.optimize.minimize
        def objective(alpha):
            return 0.5 * np.dot(alpha, np.dot(P, alpha)) - np.sum(alpha)
        
        constraints = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)}
        result = minimize(objective, np.zeros(n_samples), bounds=[(0, self.C)] * n_samples, constraints=constraints)
        self.alpha = result.x
        self.support_vectors_idx = np.where(self.alpha > 1e-5)[0]
        self.support_vectors = X[self.support_vectors_idx]
        self.support_vector_labels = y[self.support_vectors_idx]
        self.b = np.mean(self.support_vector_labels - np.sum(self.alpha * self.support_vector_labels * K[self.support_vectors_idx][:, self.support_vectors_idx], axis=1))

    def predict(self, X):
        K = self.rbf_kernel(X, self.support_vectors)
        predictions = np.dot(K, self.alpha[self.support_vectors_idx] * self.support_vector_labels) + self.b
        return np.sign(predictions)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy

def load_from_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

# Load MNIST data
dir = 'pkl_files'
X_train = load_from_pickle(f"{dir}/train_images.pkl")
Y_train = load_from_pickle(f"{dir}/train_labels.pkl")
X_test = load_from_pickle(f"{dir}/test_images.pkl")
Y_test = load_from_pickle(f"{dir}/test_labels.pkl")

# Initialize and train the RBF SVM
svm = RBF_SVM(C=1.0, gamma=0.1)
svm.fit(X_train, Y_train)

print("Fitted our data")

# Evaluate the model
train_accuracy = svm.evaluate(X_train, Y_train)
print("Evaluation finished")
test_accuracy = svm.evaluate(X_test, Y_test)
print(f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
