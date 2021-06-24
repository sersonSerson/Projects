import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm_notebook
from tensorflow.keras.layers import Dense, Dropout, Activation, \
    BatchNormalization, GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
import tensorflow as tf
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, accuracy_score
from tensorflow.keras.regularizers import L1, L2, L1L2
from tensorflow.keras.constraints import UnitNorm, MaxNorm, MinMaxNorm


def autolabel(rects, ax, decimals=4):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        value = round(height, decimals)
        ax.annotate('{}'.format(value),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', size=20)


class AnnGridSearch:
    def __init__(self, grid=None, problem='Regression', greater_is_better=False):
        self.problem = problem
        self.greater_is_better = greater_is_better
        self.default_params = self.default_parameters()
        self.grid = self.scoring_grid(grid)

    def default_parameters(self):
        default_params = \
            {
                # Configuration
                'Units': 20, 'Layers': 2, 'HiddenActivation': 'relu',
                # Early stopping
                'EarlyStopping': False, 'EarlyStoppingMetric': 'loss',
                'Patience': 100,
                # Fitting
                'BatchSize': 32, 'Epochs': 10, 'Optimizer': 'adam',
                # Activity regularization
                'ActivityRegularization': None,
                'ActivityRegularizationTime': None,
                # Batch normalization
                'BatchNormalizationTime': None,
                # Dropout
                'Dropout': False,
                # Initializer
                'Initializer': None,
                # Noise:
                'NoiseTime': None, 'Noise': None,
                # WeightsRegularization
                'WeightsRegularization': None,
                # WeightsConstraints
                'WeightsConstraints': None,
                # Scaler
                'Scaler': None,
                # ModelCheckpoint
                'ModelCheckpoint': None,
                'ModelCheckpointMetric': None
            }

        if self.problem == 'Regression':
            default_params['OutputActivation'] = 'linear'
            default_params['Loss'] = 'mean_squared_error'
            default_params['ScoreFunc'] = mean_squared_error
            default_params['Metrics'] = ['mean_squared_error']
        elif self.problem == 'BinaryClassification':
            default_params['OutputActivation'] = 'sigmoid'
            default_params['Loss'] = 'binary_crossentropy'
            default_params['ScoreFunc'] = accuracy_score
            default_params['Metrics'] = ['accuracy']
        elif self.problem == 'MultilabelClassification':
            default_params['OutputActivation'] = 'softmax'
            default_params['Loss'] = 'categorical_crossentropy'
            default_params['ScoreFunc'] = accuracy_score
            default_params['Metrics'] = ['accuracy']

        return default_params

    def scoring_grid(self, input_grid):

        def convert_name(value):
            if isinstance(value, (L1, L2, L1L2)):
                if isinstance(value, L1):
                    name = 'L1 ' + str(value.l1.reshape(-1)[0])
                elif isinstance(value, L2):
                    name = 'L2 ' + str(value.l2.reshape(-1)[0])
                elif isinstance(value, L1L2):
                    name = 'L1L2 ' + str(value.l1.reshape(-1)[0]) + ' ' +\
                           str(value.l2.reshape(-1)[0])
            elif isinstance(value, Dropout):
                name = 'Dropout ' + str(value.rate)
            elif isinstance(value, GaussianNoise):
                name = 'Noise ' + str(value.stddev)
            elif isinstance(value, (UnitNorm, MaxNorm, MinMaxNorm)):
                if isinstance(value, UnitNorm):
                    name = 'UnitNorm'
                elif isinstance(value, MaxNorm):
                    name = 'MaxNorm(' + str(value.max_value) + ')'
                elif isinstance(value, MinMaxNorm):
                    name = 'MinMaxNorm(' + str(value.min_value) + ',' + \
                           str(value.max_value) + ')'
            else:
                name = value
            return name

        if input_grid is None:
            input_grid = dict()
        # Check for misspelled keys in grid
        incorrect_keys = [key for key in input_grid.keys() if
                          key not in self.default_params.keys()]
        if len(incorrect_keys) > 0:
            raise ValueError(f'Incorrect keys provided: {incorrect_keys}')

        # Define parameters with only one possible value
        single_values = \
            [name for name, value in input_grid.items() if len(value) == 1]
        default_values = [key for key in self.default_params.keys() if
                          key not in input_grid.keys()]
        varying_values = [key for key in input_grid.keys() if
                          key not in single_values
                          and key not in default_values]
        if input_grid is None:
            input_grid = dict()
        scoring_grid = dict()

        grid = ParameterGrid(input_grid)
        for grid_combination in grid:
            name_dict = {}
            for param_name in varying_values:
                value = grid_combination[param_name]
                name_dict[param_name] = convert_name(value)

            model_name \
                = '\n'.join([f'{param_name}: {param_value}'
                            for param_name, param_value in name_dict.items()])

            if model_name == '':
                model_name = 'Default'

            for default_value in default_values:
                grid_combination[default_value] = \
                    self.default_params[default_value]
            scoring_grid[model_name] = grid_combination

        return scoring_grid

    @staticmethod
    def create_ann(params):
        tf.random.set_seed(0)
        model = Sequential()
        if params['BatchNormalizationTime'] == 'Input':
            model.add(BatchNormalization())

        hidden_layers_kwargs = \
            {'activity_regularizer': params['ActivityRegularization'],
             'kernel_initializer': params['Initializer'],
             'kernel_regularizer': params['WeightsRegularization'],
             'kernel_constraint': params['WeightsConstraints']}

        for _ in range(params['Layers']):
            if params['ActivityRegularizationTime'] == 'Before' or \
                    params['BatchNormalizationTime'] == 'Before' or \
                    params['NoiseTime'] == 'Before':

                model.add(Dense(params['Units'], activation='linear',
                               **hidden_layers_kwargs))
                if params['NoiseTime'] == 'Before' and params['Noise']:
                    model.add(params['Noise'])
                if params['BatchNormalizationTime'] == 'Before':
                    model.add(BatchNormalization())
                model.add(Activation(params['HiddenActivation']))

            else:
                model.add(Dense(params['Units'],
                                activation=params['HiddenActivation'],
                                **hidden_layers_kwargs))
                if params['NoiseTime'] == 'After' and params['Noise']:
                    model.add(params['Noise'])

                if params['BatchNormalizationTime'] == 'After':
                    model.add(BatchNormalization())

            if params['Dropout']:
                model.add(params['Dropout'])

        model.add(Dense(units=1, activation=params['OutputActivation']))
        model.compile(optimizer=params['Optimizer'],
                      loss=params['Loss'], metrics=params['Metrics'])

        return model

    def fit(self, X_train=None, y_train=None, X_test=None, y_test=None):
        score_df = pd.DataFrame(columns=['ModelName', 'ModelParams', 'Model',
                                         'History', 'Prediction', 'Score'])
        tf.random.set_seed(0)

        if X_test is not None and y_test is not None:
            validation_data = (X_test, y_test)
        else:
            validation_data = None

        for model_name, model_params in tqdm_notebook(self.grid.items()):
            callbacks = []
            if model_params['EarlyStopping']:
                monitor = model_params['EarlyStoppingMetric']
                if validation_data:
                    monitor = 'val_' + monitor
                es = EarlyStopping(monitor=monitor, mode='min', verbose=1,
                                   patience=model_params['Patience'])
                callbacks.append(es)
            if model_params['ModelCheckpoint']:
                monitor = model_params['ModelCheckpointMetric']
                mode = 'max' if self.greater_is_better else 'min'
                mc = ModelCheckpoint(f'{model_name}.h5',
                                     monitor=monitor, mode=mode,
                                     verbose=0, save_best_only=True)
                callbacks.append(mc)
            model = self.create_ann(model_params)
            history = model.fit(X_train, y_train, epochs=model_params['Epochs'],
                                validation_data=validation_data,
                                batch_size=model_params['BatchSize'],
                                verbose=0,
                                callbacks=callbacks)
            score_dict = {'Model': model,
                          'ModelName': model_name,
                          'History': history,
                          'ModelParams': model_params}
            score_df = score_df.append(score_dict, ignore_index=True)
        self.score_df = score_df

    def score(self, X_test=None, y_test=None):
        tf.random.set_seed(0)
        score_df = self.score_df
        for i in range(len(score_df)):
            test_pred = score_df.loc[i, 'Model'].predict(X_test, verbose=0)
            scaler = score_df.loc[i, 'ModelParams']['Scaler']
            if scaler:
                test_pred = \
                    scaler.inverse_transform(test_pred.reshape(-1, 1)).reshape(-1, 1)
                y_test = \
                    scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1, 1)
            score_function = score_df.loc[i, 'ModelParams']['ScoreFunc']
            score_df.loc[i, 'Score'] = score_function(y_test, test_pred)
            score_df.loc[i, 'Prediction'] = test_pred
        self.score_df = score_df.sort_values('Score',
                                             ascending=not self.greater_is_better)
        return self.score_df[['ModelName', 'Score']]

    def score_bar(self, n_results=5):
        fig, ax = plt.subplots(figsize=(2 * n_results, 5))
        rects = ax.bar(self.score_df['ModelName'][:n_results],
                       self.score_df['Score'][:n_results])
        autolabel(rects, ax)
        plt.xticks(rotation=90)
        bottom, top = plt.ylim()  # return the current ylim
        plt.ylim(top=top * 1.1)
        plt.title('Scores', size=20)
        plt.tight_layout()
        plt.show()

    def leaning_curve(self, train_metric=None, val_metric=None,
                      strarting_epoch=0, n_results=5):
        score_df = self.score_df
        if n_results > len(score_df):
            n_results = len(score_df)

        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(n_results):
            model_name = score_df.iloc[i]['ModelName']
            if train_metric:
                history = score_df.iloc[i]['History'].history[train_metric]
                ax.plot(history[strarting_epoch:],
                        label=f'Train: {model_name}')
            if val_metric:
                history = score_df.iloc[i]['History'].history[val_metric]
                ax.plot(history[strarting_epoch:],
                        label=f'Validation: {model_name}')
        ax.set_xlim(xmin=strarting_epoch)
        plt.title('Learning curve', size=20)
        plt.legend()
        plt.show()

    def best_prediction(self):
        return np.array(self.score_df.iloc[0]['Prediction'])