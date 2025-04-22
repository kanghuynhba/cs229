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

    # *** START CODE HERE ***

    # Training logistic regression
    model = LogisticRegression(eps=1e-5)
    model.fit(x_train, y_train)

    # Plot data and decision boundary
    util.plot(x_train, y_train, model.theta, 'output/p01b_{}.png'.format(pred_path[-5]))
    
    # Save predictions
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred>0.5, fmt='%d')
    
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

        # # Newton's ethod
        # while True:
        #     old_theta=self.theta

        #     # Compute Hypothesis, Gradient & Hessian Matrix
        #     h_x=1/(1+np.exp(-x.dot(self.theta)))
        #     gradient_J_theta=x.T.(h_x-y)/m
        #     hessian=(x.T*h_x*(1-h_x)).dot(x)/m

        #     # Update theta
        #     self.theta-= np.linalg.inv(hessian).dot(gradient_J_theta)

        #     # End training (break condition)
        #     if np.linalg.norm(self.theta-old_theta, ord=1)<self.eps:
        #         break

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """


        # *** START CODE HERE ***
        # return h(self.theta, x)>=0.5
        return 1/(1+np.exp(-x.dot(self.theta)))
        # *** END CODE HERE ***
