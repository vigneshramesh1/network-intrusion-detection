import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
import scipy
import sklearn
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
import torchmetrics
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from sklearn.metrics import confusion_matrix
from skorch import NeuralNetClassifier

# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Torch', torch.__version__, 'CUDA', torch.version.cuda)
print('Device:', torch.device('cuda:0'))

# number of subprocesses to use for data loading
num_workers = multiprocessing.cpu_count()

# how many samples per batch to load
batch_size = 64

# percentage of data set to use as validation
valid_size = 0.15

df = pd.read_csv('/scratch/vrames25/NIDS/datasets/csv_preprocessed.csv', low_memory=False)
print(df.shape)

num_classes = df['Attack Type'].nunique()
print(num_classes)

# Train Test split
X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'Attack Type'], df['Attack Type'],
                                                    stratify=df['Attack Type'],
                                                    shuffle=True,
                                                    test_size=0.15)

X_train = pd.DataFrame(X_train, columns=df.drop(columns=['Attack Type']).columns.to_list())
X_test = pd.DataFrame(X_test, columns=df.drop(columns=['Attack Type']).columns.to_list())
y_train = pd.DataFrame(y_train, columns=['Attack Type'])
y_test = pd.DataFrame(y_test, columns=['Attack Type'])

print("Training dataset size:", X_train.shape)
print("Testing dataset size:", X_test.shape)
print("Training target size:", y_train.shape)
print("Testing target size:", y_test.shape)

# Number of features
num_features = X_train.shape[1]

# Creating a PyTorch class
# input_features ==> 12 ==> 32
class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Building a linear encoder with Linear
        # layer followed by Tanh activation function
        # input_features ===> 12
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(num_features, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 12),
            torch.nn.Tanh()
        )

        # Dense neural network layers
        self.dense_nn = torch.nn.Sequential(
            torch.nn.Linear(12, 32),  # Input size is 12 from the encoder
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, num_classes),  # Output size is the number of classes
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        output = self.dense_nn(encoded)
        return output

param_grid = {'optimizer__lr': [0.01, 0.001, 0.0001],
              'max_epochs': [4, 8, 16, 32],
              'batch_size':[1, 32, 64, 128, 256, 512, 1024]}

model = NeuralNetClassifier(
    module=Autoencoder,
    max_epochs=4,
    batch_size=32,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    verbose=True,
    device=device  # Specify device
)

print(model.initialize())

# Use StratifiedKFold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,
                           cv=skf,
                           scoring='accuracy', verbose=10)
grid_search.fit(X_train.values.astype(np.float32), y_train.values.ravel())

# Get the best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)