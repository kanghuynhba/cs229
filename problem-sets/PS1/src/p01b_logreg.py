import numpy as np
import matplotlib.pyplot as plt
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***

    # plt.xlabel('x1')
    # plt.ylabel('x2')
    # plt.plot(x_train[y_train==1, 1], x_train[y_train==1, 2], 'bx', linewidth=2)
    # plt.plot(x_train[y_train==0, 1], x_train[y_train==0, 2], 'go', linewidth=2)

    clf=LogisticRegression()
    clf.fit(x_train, y_train)
    print(clf.predict(x_eval))
    print(y_eval)
    
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
            
    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """

        # *** START CODE HERE ***

        def h(theta, x):
            return 1/(1+np.exp(-x.dot(theta))); # return Shape (m,) 

        def gradient(theta , x, y):
            m, _ = x.shape
            return -1/m*np.dot(x.T, (y-h(theta, x))) # return Shape (n,)
            
        def hessian(theta, x):
            m, _ = x.shape
            h_theta_x = np.reshape(h(theta, x), (-1, 1)) # reshape (m,) => (m, 1)
            return 1/m*np.dot(x.T, h_theta_x*(1-h_theta_x)*x) # return Shape (n, n)

        def next_theta(theta, x, y):
            return theta-np.linalg.inv(hessian(theta, x)).dot(gradient(theta, x, y)) # return Shape (n,)
        
        m, n = x.shape
        if self.theta is None:
            self.theta=np.zeros(n)

        old_theta=self.theta
        new_theta=next_theta(self.theta, x, y)
        while np.linalg.norm(new_theta-old_theta, 1) >= self.eps:
            old_theta = new_theta
            new_theta = next_theta(old_theta, x, y)
        self.theta=new_theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """

        # *** START CODE HERE ***
        return x.dot(self.theta)>=0
        # *** END CODE HERE ***
