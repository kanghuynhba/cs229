import numpy as np
import math
import util
import matplotlib.pyplot as plt

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    # The line below is the original one from Stanford. It does not include the intercept, but this should be added.
    # x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path

    # Training poisson regression
    model=PoissonRegression(step_size= lr,eps=1e-5)
    model.fit(x_train, y_train)

    
    # Save predictions
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred=np.round(model.predict(x_eval))
    np.savetxt(pred_path, np.column_stack((np.round(y_pred), y_eval)))

    plt.figure()
    plt.plot(y_eval, y_pred, 'bx')
    plt.xlabel('true counts')
    plt.ylabel('predict counts')
    plt.savefig('output/p03.png')

    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def h(self, theta, x):
        return np.exp(x.dot(theta)) # return Shape (m,)

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***

        m, n = x.shape
        
        # derivative of log-likelihood
        def gradient(theta):
            return np.dot(x.T, (y-self.h(theta, x))) # return Shape (n,)

        def next_step(theta):
            return self.step_size/ m * gradient(theta) # return Shape (n,)
        
        if self.theta is None:
            self.theta=np.zeros(n)

        while True:
            theta = np.copy(self.theta)
            self.theta += next_step(theta)
            if np.linalg.norm(self.theta-theta, ord=1) < self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return self.h(self.theta, x) 
        # *** END CODE HERE ***
