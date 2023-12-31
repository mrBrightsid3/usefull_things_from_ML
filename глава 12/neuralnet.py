import numpy as np
import sys


class NeuralNetMLP(object):
    """Нейронная сеть прямого распространения / классификатор на основе многослойного перцептрона.

    Parameters
    ------------
    n_hidden : int (default: 30)
        Количество скрытых элементов.
    l2 : float (default: 0.)
        Значение лямбда для регуляризации l2
        регуляризация отсутствует, если l2=0. (default)
    epochs : int (default: 100)
        Эпохи
    eta : float (default: 0.001)
        Скорость обучения
    shuffle : bool (default: True)
        Тасовать ли данные во избежании циклов
    minibatch_size : int (default: 1)
        Количество обучающих образцов в минипакете
    seed : int (default: None)


    Attributes
    -----------
    eval_ : dict
      словарь, в котором собираются показатели издержек и правильности
      при обучении и правильности при испытании для каждой эпохи

    """

    def __init__(
        self,
        n_hidden=30,
        l2=0.0,
        epochs=100,
        eta=0.001,
        shuffle=True,
        minibatch_size=1,
        seed=None,
    ):
        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _onehot(self, y, n_classes):
        """кодирует метки в представление с унитарным кодом

        Parameters
        ------------
        y : array, shape = [n_examples]
            целевые значения
        n_classes : int
            количество классов

        Returns
        -----------
        onehot : array, shape = (n_examples, n_labels)

        """
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.0
        return onehot.T

    def _sigmoid(self, z):
        """Compute logistic function (sigmoid)"""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def _forward(self, X):
        """Вычисляет шаг прямого распространения"""

        # шаг 1: общий вход скрытого слоя
        # скалярное произведение [n_examples, n_features] и [n_features, n_hidden]
        # -> [n_examples, n_hidden]
        z_h = np.dot(X, self.w_h) + self.b_h

        # шаг 2: активация скрытого слоя
        a_h = self._sigmoid(z_h)

        # шаг 3: net input of output layer
        # скалярное произведение [n_examples, n_hidden] и [n_hidden, n_classlabels]
        # -> [n_examples, n_classlabels]

        z_out = np.dot(a_h, self.w_out) + self.b_out

        # шаг 4: активация выходного слоя
        a_out = self._sigmoid(z_out)

        return z_h, a_h, z_out, a_out

    def _compute_cost(self, y_enc, output):
        """вычисление логистической функции издержек.

        Parameters
        ----------
        y_enc : array, shape = (n_examples, n_labels)
            метки классов в унитарном коде
        output : array, shape = [n_examples, n_output_units]
            активация выходного слоя

        Returns
        ---------
        cost : float
            Регуляризованные издержки

        """
        L2_term = self.l2 * (
            np.sum(self.w_h**2.0) + np.sum(self.w_out**2.0)
        )  # член регуляризации

        term1 = -y_enc * (np.log(output))
        term2 = (1.0 - y_enc) * np.log(1.0 - output)
        cost = np.sum(term1 - term2) + L2_term

        # If you are applying this cost function to other
        # datasets where activation
        # values maybe become more extreme (closer to zero or 1)
        # you may encounter "ZeroDivisionError"s due to numerical
        # instabilities in Python & NumPy for the current implementation.
        # I.e., the code tries to evaluate log(0), which is undefined.
        # To address this issue, you could add a small constant to the
        # activation values that are passed to the log function.
        #
        # For example:
        #
        # term1 = -y_enc * (np.log(output + 1e-5))
        # term2 = (1. - y_enc) * np.log(1. - output + 1e-5)

        return cost

    def predict(self, X):
        """Predict class labels

        Parameters
        -----------
        X : array, shape = [n_examples, n_features]
            входной слой с первоначальными признаками

        Returns:
        ----------
        y_pred : array, shape = [n_examples]
            предсказанные метки классов

        """
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred

    def fit(self, X_train, y_train, X_valid, y_valid):
        """Learn weights from training data.

        Parameters
        -----------
        X_train : array, shape = [n_examples, n_features]
            Input layer with original features.
        y_train : array, shape = [n_examples]
            Target class labels.
        X_valid : array, shape = [n_examples, n_features]
            Sample features for validation during training
        y_valid : array, shape = [n_examples]
            Sample labels for validation during training

        Returns:
        ----------
        self

        """
        n_output = np.unique(y_train).shape[0]  # number of class labels
        n_features = X_train.shape[1]

        ########################
        # Weight initialization
        ########################

        # weights for input -> hidden
        self.b_h = np.zeros(self.n_hidden)
        self.w_h = self.random.normal(
            loc=0.0, scale=0.1, size=(n_features, self.n_hidden)
        )

        # weights for hidden -> output
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(
            loc=0.0, scale=0.1, size=(self.n_hidden, n_output)
        )

        epoch_strlen = len(str(self.epochs))  # for progress formatting
        self.eval_ = {"cost": [], "train_acc": [], "valid_acc": []}

        y_train_enc = self._onehot(y_train, n_output)

        # iterate over training epochs
        for i in range(self.epochs):
            # iterate over minibatches
            indices = np.arange(X_train.shape[0])

            if self.shuffle:
                self.random.shuffle(indices)

            for start_idx in range(
                0, indices.shape[0] - self.minibatch_size + 1, self.minibatch_size
            ):
                batch_idx = indices[start_idx : start_idx + self.minibatch_size]

                # forward propagation
                z_h, a_h, z_out, a_out = self._forward(X_train[batch_idx])

                ##################
                # Backpropagation
                ##################

                # [n_examples, n_classlabels]
                delta_out = a_out - y_train_enc[batch_idx]

                # [n_examples, n_hidden]
                sigmoid_derivative_h = a_h * (
                    1.0 - a_h
                )  # вычисление производной сигмоидальной функции

                # [n_examples, n_classlabels] dot [n_classlabels, n_hidden]
                # -> [n_examples, n_hidden]
                delta_h = (
                    np.dot(delta_out, self.w_out.T) * sigmoid_derivative_h
                )  # умнижаем ошибку на выходные веса и на производную

                # [n_features, n_examples] dot [n_examples, n_hidden]
                # -> [n_features, n_hidden]
                grad_w_h = np.dot(X_train[batch_idx].T, delta_h)
                grad_b_h = np.sum(delta_h, axis=0)

                # [n_hidden, n_examples] dot [n_examples, n_classlabels]
                # -> [n_hidden, n_classlabels]
                grad_w_out = np.dot(a_h.T, delta_out)
                grad_b_out = np.sum(delta_out, axis=0)

                # обновляем веса с регуляризацией
                delta_w_h = grad_w_h + self.l2 * self.w_h
                delta_b_h = grad_b_h  # член смещения не регуляризуется
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h

                delta_w_out = grad_w_out + self.l2 * self.w_out
                delta_b_out = grad_b_out  # bias is not regularized
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out

            #############
            # Evaluation
            #############

            # Evaluation after each epoch during training
            z_h, a_h, z_out, a_out = self._forward(X_train)

            cost = self._compute_cost(y_enc=y_train_enc, output=a_out)

            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)

            train_acc = (np.sum(y_train == y_train_pred)).astype(
                np.float
            ) / X_train.shape[0]
            valid_acc = (np.sum(y_valid == y_valid_pred)).astype(
                np.float
            ) / X_valid.shape[0]

            sys.stderr.write(
                "\r%0*d/%d | Cost: %.2f "
                "| Train/Valid Acc.: %.2f%%/%.2f%% "
                % (
                    epoch_strlen,
                    i + 1,
                    self.epochs,
                    cost,
                    train_acc * 100,
                    valid_acc * 100,
                )
            )
            sys.stderr.flush()

            self.eval_["cost"].append(cost)
            self.eval_["train_acc"].append(train_acc)
            self.eval_["valid_acc"].append(valid_acc)

        return self
