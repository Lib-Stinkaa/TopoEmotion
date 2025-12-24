"""Shared configuration for emotion recognition experiments"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Random seed
RANDOM_STATE = 42

# Parallel processing
N_PARALLEL_JOBS = 15

# Emotion thresholds
VALENCE_THRESHOLD = 5.0
AROUSAL_THRESHOLD = 5.0

# Signal types
SIGNAL_TYPES = [
    'ecg_clean', 'bvp_clean', 'gsr_clean', 'rsp_clean',
    'skt_clean', 'emg_zygo_clean', 'emg_coru_clean', 'emg_trap_clean'
]

# 10 Classifiers
CLASSIFIERS = {
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10,
                                          random_state=RANDOM_STATE, n_jobs=-1),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=100, max_depth=10,
                                      random_state=RANDOM_STATE, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                   random_state=RANDOM_STATE),
    'LogisticRegression': LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, n_jobs=-1),
    'Ridge': RidgeClassifier(random_state=RANDOM_STATE),
    'SVM': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE, cache_size=500),
    'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'DecisionTree': DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
    'NaiveBayes': GaussianNB(),
    'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300,
                        random_state=RANDOM_STATE, early_stopping=True)
}
