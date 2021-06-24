from unittest import TestCase
from ann.ANNGrid import AnnGridSearch
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras.layers import Dropout, GaussianNoise


class TestAnnGridSearch(TestCase):
    def test_ann_grid_search(self):
        # Can create grid search class
        AnnGridSearch()

    def test_default_params(self):
        # Can create default parameters dictionary
        default_params = AnnGridSearch().default_parameters()
        print(default_params)

    def test_scoring_grid(self):
        # Can create grid that will be used for scoring (initial grid +
        # default parameters)
        scoring_grid = AnnGridSearch().grid
        print(scoring_grid)

    def test_create_ann(self):
        # Can create NN with default params
        grid_search = AnnGridSearch()
        default_params = grid_search.default_parameters()
        grid_search.create_ann(default_params)


class TestAnnGridSearchRegression(TestCase):

    def setUp(self):
        X, y = make_regression(n_samples=100, n_features=2, bias=0.1,
                               random_state=0)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.2, random_state=0)

    def test_score(self):
        # Can score NN on test data
        grid_search = AnnGridSearch()
        grid_search.fit(self.X_train, self.y_train, self.X_test, self.y_test)
        score_df = grid_search.score(self.X_test, self.y_test)
        print(score_df)

    def test_score_bar(self):
        # Can score NN on test data
        grid_search = AnnGridSearch()
        grid_search.fit(self.X_train, self.y_train, self.X_test, self.y_test)
        grid_search.score(self.X_test, self.y_test)
        grid_search.score_bar()

    def test_leaning_curve(self):
        # Can score NN on test data
        grid_search = AnnGridSearch()
        grid_search.fit(self.X_train, self.y_train, self.X_test, self.y_test)
        grid_search.leaning_curve(metric='val_loss', strarting_epoch=0)


class Test(TestAnnGridSearchRegression):

    def fit(self, grid):
        grid_search = AnnGridSearch(grid)
        grid_search.fit(self.X_train, self.y_train, self.X_test, self.y_test)

    def test_structure(self):
        grid = {'Units': [10, 20, 30], 'Layers': [1, 2], 'Epochs': [10]}
        self.fit(grid)

    def test_activity_regularization(self):
        grid = {'Units': [20], 'Layers': [2], 'Epochs': [10],
                'ActivityRegularizationTime': ['Before', 'After'],
                'ActivityRegularization': [l2(0.1), l1(0.1), l1_l2(0.1, 0.1)]}
        self.fit(grid)

    def test_batch_normalization(self):
        grid = {'Units': [10], 'Layers': [1], 'Epochs': [10],
                'BatchNormalizationTime': ['Before', 'After', 'Input', None]}
        self.fit(grid)

    def test_batch_size(self):
        grid = {'Units': [10], 'Layers': [1], 'Epochs': [10],
                'BatchSize': [1, 3, 5, 10, 32, 50]}
        self.fit(grid)

    def test_dropout(self):
        grid = {'Units': [10], 'Layers': [1], 'Epochs': [10],
                'DropOut': [Dropout(0.1), Dropout(0.2)]}
        self.fit(grid)

    def test_early_stopping(self):
        grid = {'Units': [10], 'Layers': [1], 'Epochs': [100],
                'EarlyStopping': [True], 'Patience': [20, 50, 70]}
        self.fit(grid)

    def test_initializers(self):
        grid = {'Units': [10], 'Layers': [1], 'Epochs': [10],
                'Initializer': ['glorot_normal', 'glorot_uniform']}
        self.fit(grid)

    def test_optimizers(self):
        grid = {'Units': [10], 'Layers': [1], 'Epochs': [10],
                'Optimizer': ['adam', 'rmsprop']}
        self.fit(grid)

    def test_loss(self):
        grid = {'Units': [10], 'Layers': [1], 'Epochs': [10],
                'Loss': ['mean_squared_error',
                         'mean_squared_logarithmic_error']}
        self.fit(grid)

    def test_noise(self):
        grid = {'Units': [10], 'Layers': [1], 'Epochs': [10],
                'Noise': [GaussianNoise(0.05)],
                'NoiseTime': ['Before', 'After']}
        self.fit(grid)

    def test_weight_regularization(self):
        grid = {'Units': [10], 'Layers': [1], 'Epochs': [10],
                'WeightsRegularization': [l2(0.1),  l1(0.1), l1_l2(0.1, 0.1)]}
        self.fit(grid)

    def test_incorrect_keys(self):
        grid = {'Unitsss': [10]}
        self.assertRaises(ValueError, self.fit, grid=grid)
