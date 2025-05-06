import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

df = pd.read_csv('training_data.csv')
target = pd.read_csv('training_data_targets.csv',names = ['label'])


df_encoded = pd.get_dummies(df, columns=['hypertensive', 'atrialfibrillation', 'CHD with no MI', 'diabetes', 'deficiencyanemias', 'depression', 'Hyperlipemia', 'Renal failure', 'COPD'], prefix=('hypertensive', 'atrialfibrillation', 'CHD with no MI', 'diabetes', 'deficiencyanemias', 'depression', 'Hyperlipemia', 'Renal failure', 'COPD'))                          
print(df_encoded)

df_attach = df_encoded.copy()
df_attach['Y'] = target['label']

df_attach_pure = df_attach.dropna()
target_pure = df_attach_pure['Y']
df_pure = df_attach_pure.drop('Y',axis = 1)

row = df.index
clm= df.columns

h = df.isnull().sum(axis=0)
print(h)

df_plain = df_encoded.copy()
dropcol= []
for i in range(len(h)):
    if h[i]>=211:
        dropcol.append(df_plain.columns[i])

df_twenty= df_plain.drop(dropcol,axis = 1)
df_twenty

dropcol


# ## NaN value Imputation

# Mean Value Imputation where no column is dropped


df_nodrop_mean = df_encoded.copy()
for column in df_nodrop_mean.columns:
    mean_nodrop = df_nodrop_mean[column].mean()
    df_nodrop_mean[column].fillna(mean_nodrop, inplace = True)


k = df_nodrop_mean.isnull().sum(axis=0)
k


# Mean Value Imputation where column with more than 20% value missing dropped


df_twenty_mean = df_twenty
for column in df_twenty_mean.columns:
    mean_twenty = df_twenty_mean[column].mean()
    df_twenty_mean[column].fillna(mean_twenty, inplace = True)

h = df_twenty_mean.isnull().sum(axis=0)

# Median Value Imputation where no column is dropped

df_nodrop_median= df_encoded.copy()
for column in df_nodrop_median.columns:
    median_nodrop = df_nodrop_median[column].median()
    df_nodrop_median[column].fillna(median_nodrop, inplace = True)


# Median Value Imputation where column with more than 20% value missing dropped

df_twenty_median= df_twenty.copy()
for column in df_twenty_median.columns:
    median_twenty = df_twenty_median[column].median()
    df_twenty_median[column].fillna(median_twenty, inplace = True)


# (Mean + Standard Deviation) Value Imputation where no column is dropped

df_nodrop_mpstd= df_encoded.copy()
for column in df_nodrop_mpstd.columns:
    mean_nodrop_mpstd = df_nodrop_mpstd[column].mean()
    std_nodrop_mpstd = df_nodrop_mpstd[column].std()
    f = mean_nodrop_mpstd + std_nodrop_mpstd
    df_nodrop_mpstd[column].fillna(f, inplace=True)


# (Mean + Standard Deviation) Value Imputation where column with more than 20% value missing dropped 


df_twenty_mpstd= df_twenty.copy()
for column in df_twenty_mpstd.columns:
    mean_twenty_mpstd = df_twenty_mpstd[column].mean()
    std_twenty_mpstd = df_twenty_mpstd[column].std()
    g =  mean_twenty_mpstd + std_twenty_mpstd 
    df_twenty_mpstd[column].fillna(g, inplace=True)


# (Mean - Standard Deviation) Value Imputation where no column is dropped


df_nodrop_mmstd= df_encoded.copy()
for column in df_nodrop_mmstd.columns:
    mean_nodrop_mmstd = df_nodrop_mmstd[column].mean()
    std_nodrop_mmstd = df_nodrop_mmstd[column].std()
    df_nodrop_mmstd[column].fillna(mean_nodrop_mmstd + std_nodrop_mmstd, inplace=True)


# (Mean - Standard Deviation) Value Imputation where column with more than 20% value missing dropped


df_twenty_mmstd= df_twenty.copy()
for column in df_twenty_mmstd.columns:
    mean_twenty_mmstd = df_twenty_mmstd[column].mean()
    std_twenty_mmstd = df_twenty_mmstd[column].std()
    df_twenty_mmstd[column].fillna(mean_twenty_mmstd - std_twenty_mmstd, inplace=True)


# KNN Imputation where no column is dropped


from sklearn.impute import KNNImputer
df_nodrop_knn = df_encoded.copy()
for column in df_nodrop_knn.columns:
    knn_nodrop_imputer = KNNImputer(n_neighbors=3)
    col_nodrop_values = df_nodrop_knn[column].values.reshape(-1, 1)
    col_nodrop_imputed = knn_nodrop_imputer.fit_transform(col_nodrop_values)
    df_nodrop_knn[column] = col_nodrop_imputed


# KNN Imputation where column with more than 20% value missing dropped 

df_twenty_knn = df_twenty.copy()
for column in df_twenty_knn.columns:
    knn_twenty_imputer = KNNImputer(n_neighbors=3)
    col_twenty_values = df_twenty_knn[column].values.reshape(-1, 1)
    col_twenty_imputed = knn_twenty_imputer.fit_transform(col_twenty_values)
    df_twenty_knn[column] = col_twenty_imputed


# ## Importing Necessary Libraries

from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


# ## Running different models

# Running the model after Mean Imputation without dropping any column

X = df_nodrop_mean
Y = target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify = Y)


# Create a pipeline dictionary with models and their hyperparameters
pipeline_dict = {
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'model__n_estimators': [25, 50, 100, 200],
            'model__max_depth': [10, 20, 30],
            'model__criterion':['gini', 'entropy', 'log_loss'],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2', None],
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(),
        'params': {
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l1', 'l2', 'elasticnet'],
            'model__max_iter': [100, 200, 300]
        }
    },
    
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'model__max_depth': [None, 5, 10, 20, 30],
            'model__min_samples_split': [1, 2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__criterion': ['gini', 'entropy'],
            'model__max_features': ['auto', 'sqrt', 'log2']
        }
    },
    'adaboost': {
        'model': AdaBoostClassifier(),
        'params': {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 1],
            'model__algorithm': ['SAMME', 'SAMME.R']
        }
    },
    'naive_bayes': {  
        'model': GaussianNB(),
        'params': {
            'model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
    },
    'knn': {
        'model': KNeighborsClassifier(),
        'params': {
            'model__n_neighbors': [3, 5, 7, 9],
            'model__weights': ['uniform', 'distance'],
            'model__p': [1, 2]
        }
    },
#     'svm': {
#         'model': SVC(),
#         'params': {
#             'model__C': [0.1, 1, 10],
#             'model__kernel': ['linear', 'rbf'],
#             'model__gamma': ['scale', 'auto']
#         }
#     }
}

# Create an empty list to store the best models
best_models = []

# Iterate through the pipeline dictionary and fit models using GridSearchCV
for model_name, config in pipeline_dict.items():
    print(f"\nRunning GridSearchCV for {model_name}")
    
    # Create a pipeline with a scaler (if needed) and the model
    if 'scaler' in config:
        pipeline = Pipeline([
            ('scaler', config['scaler']),
            ('model', config['model'])
        ])
    else:
        pipeline = Pipeline([
            ('model', config['model'])
        ])
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        config['params'],
        cv=5,  # 5-fold cross-validation
        scoring='f1_macro',
        n_jobs=-1  # Use all available CPUs
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and the corresponding model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    # Predict on the test set and calculate accuracy
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Best Parameters: {best_params}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    report = classification_report(y_test, y_pred, output_dict=True)
    f1_macro = report['macro avg']['f1-score']
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']

    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix for {model_name}:\n{cm}")
    
    # Save the best model
    best_models.append((model_name, best_model, accuracy, f1_macro, precision, recall))
    
    print("\nClassification Report:")
    print(classification_report(y_test, best_model.predict(X_test)))

best_models.sort(key=lambda x: (x[3]), reverse=True)
best_model_name, best_model, best_accuracy, best_f1_macro, best_precision, best_recall = best_models[0]

print(f"\nBest Model: {best_model_name}")
print(f"Best Test Accuracy: {best_accuracy:.4f}")
print(f"Best F1 Score (Macro): {best_f1_macro:.4f}")
print(f"Best Precision (Macro): {best_precision:.4f}")
print(f"Best Recall (Macro): {best_recall:.4f}")

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, best_model.predict(X_test)))


# Running the Model after Mean Value Imputation where column with more than 20% value missing dropped

X = df_twenty_mean
Y = target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=17, stratify = Y) # stratify.


# Create a pipeline dictionary with models and their hyperparameters
pipeline_dict = {
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'model__n_estimators': [25, 50, 100, 200],
            'model__max_depth': [10, 20, 30],
            'model__criterion':['gini', 'entropy', 'log_loss'],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2', None],
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(),
        'params': {
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l1', 'l2', 'elasticnet'],
            'model__max_iter': [100, 200, 300]
        }
    },
    
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'model__max_depth': [None, 5, 10, 20, 30],
            'model__min_samples_split': [1, 2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__criterion': ['gini', 'entropy'],
            'model__max_features': ['auto', 'sqrt', 'log2']
        }
    },
    'adaboost': {
        'model': AdaBoostClassifier(),
        'params': {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 1],
            'model__algorithm': ['SAMME', 'SAMME.R']
        }
    },
    'naive_bayes': {  
        'model': GaussianNB(),
        'params': {
            'model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
    },
    'knn': {
        'model': KNeighborsClassifier(),
        'params': {
            'model__n_neighbors': [3, 5, 7, 9],
            'model__weights': ['uniform', 'distance'],
            'model__p': [1, 2]
        }
    },
#     'svm': {
#         'model': SVC(),
#         'params': {
#             'model__C': [0.1, 1, 10],
#             'model__kernel': ['linear', 'rbf'],
#             'model__gamma': ['scale', 'auto']
#         }
#     }
}

# Create an empty list to store the best models
best_models = []

# Iterate through the pipeline dictionary and fit models using GridSearchCV
for model_name, config in pipeline_dict.items():
    print(f"\nRunning GridSearchCV for {model_name}")
    
    # Create a pipeline with a scaler (if needed) and the model
    if 'scaler' in config:
        pipeline = Pipeline([
            ('scaler', config['scaler']),
            ('model', config['model'])
        ])
    else:
        pipeline = Pipeline([
            ('model', config['model'])
        ])
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        config['params'],
        cv=5,  # 5-fold cross-validation
        scoring='f1_macro',
        n_jobs=-1  # Use all available CPUs
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and the corresponding model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    # Predict on the test set and calculate accuracy
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Best Parameters: {best_params}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    report = classification_report(y_test, y_pred, output_dict=True)
    f1_macro = report['macro avg']['f1-score']
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']

    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix for {model_name}:\n{cm}")

    
    # Save the best model
    best_models.append((model_name, best_model, accuracy, f1_macro, precision, recall))
    
    print("\nClassification Report:")
    print(classification_report(y_test, best_model.predict(X_test)))

best_models.sort(key=lambda x: (x[3]), reverse=True)
best_model_name, best_model, best_accuracy, best_f1_macro, best_precision, best_recall = best_models[0]

print(f"\nBest Model: {best_model_name}")
print(f"Best Test Accuracy: {best_accuracy:.4f}")
print(f"Best F1 Score (Macro): {best_f1_macro:.4f}")
print(f"Best Precision (Macro): {best_precision:.4f}")
print(f"Best Recall (Macro): {best_recall:.4f}")

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, best_model.predict(X_test)))


# Running the model after Median Imputation without dropping any column

X = df_nodrop_median
Y = target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=17, stratify = Y) # stratify.

# Create a pipeline dictionary with models and their hyperparameters
pipeline_dict = {
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'model__n_estimators': [25, 50, 100, 200],
            'model__max_depth': [10, 20, 30],
            'model__criterion':['gini', 'entropy', 'log_loss'],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2', None],
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(),
        'params': {
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l1', 'l2', 'elasticnet'],
            'model__max_iter': [100, 200, 300]
        }
    },
    
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'model__max_depth': [None, 5, 10, 20, 30],
            'model__min_samples_split': [1, 2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__criterion': ['gini', 'entropy'],
            'model__max_features': ['auto', 'sqrt', 'log2']
        }
    },
    'adaboost': {
        'model': AdaBoostClassifier(),
        'params': {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 1],
            'model__algorithm': ['SAMME', 'SAMME.R']
        }
    },
    'naive_bayes': {  
        'model': GaussianNB(),
        'params': {
            'model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
    },
    'knn': {
        'model': KNeighborsClassifier(),
        'params': {
            'model__n_neighbors': [3, 5, 7, 9],
            'model__weights': ['uniform', 'distance'],
            'model__p': [1, 2]
        }
    },
#     'svm': {
#         'model': SVC(),
#         'params': {
#             'model__C': [0.1, 1, 10],
#             'model__kernel': ['linear', 'rbf'],
#             'model__gamma': ['scale', 'auto']
#         }
#     }
}

# Create an empty list to store the best models
best_models = []

# Iterate through the pipeline dictionary and fit models using GridSearchCV
for model_name, config in pipeline_dict.items():
    print(f"\nRunning GridSearchCV for {model_name}")
    
    # Create a pipeline with a scaler (if needed) and the model
    if 'scaler' in config:
        pipeline = Pipeline([
            ('scaler', config['scaler']),
            ('model', config['model'])
        ])
    else:
        pipeline = Pipeline([
            ('model', config['model'])
        ])
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        config['params'],
        cv=5,  # 5-fold cross-validation
        scoring='f1_macro',
        n_jobs=-1  # Use all available CPUs
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and the corresponding model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    # Predict on the test set and calculate accuracy
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Best Parameters: {best_params}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    report = classification_report(y_test, y_pred, output_dict=True)
    f1_macro = report['macro avg']['f1-score']
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']

    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix for {model_name}:\n{cm}")
    
    # Save the best model
    best_models.append((model_name, best_model, accuracy, f1_macro, precision, recall))
    
    print("\nClassification Report:")
    print(classification_report(y_test, best_model.predict(X_test)))

best_models.sort(key=lambda x: (x[3]), reverse=True)
best_model_name, best_model, best_accuracy, best_f1_macro, best_precision, best_recall = best_models[0]

print(f"\nBest Model: {best_model_name}")
print(f"Best Test Accuracy: {best_accuracy:.4f}")
print(f"Best F1 Score (Macro): {best_f1_macro:.4f}")
print(f"Best Precision (Macro): {best_precision:.4f}")
print(f"Best Recall (Macro): {best_recall:.4f}")

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, best_model.predict(X_test)))


# Running the model after after Median Imputation by dropping columns where more than 20% values are missing


X = df_twenty_median
Y = target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=17, stratify = Y) # stratify.


# In[102]:


# Create a pipeline dictionary with models and their hyperparameters
pipeline_dict = {
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'model__n_estimators': [25, 50, 100, 200],
            'model__max_depth': [10, 20, 30],
            'model__criterion':['gini', 'entropy', 'log_loss'],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2', None],
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(),
        'params': {
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l1', 'l2', 'elasticnet'],
            'model__max_iter': [100, 200, 300]
        }
    },
    
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'model__max_depth': [None, 5, 10, 20, 30],
            'model__min_samples_split': [1, 2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__criterion': ['gini', 'entropy'],
            'model__max_features': ['auto', 'sqrt', 'log2']
        }
    },
    'adaboost': {
        'model': AdaBoostClassifier(),
        'params': {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 1],
            'model__algorithm': ['SAMME', 'SAMME.R']
        }
    },
    'naive_bayes': {  
        'model': GaussianNB(),
        'params': {
            'model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
    },
    'knn': {
        'model': KNeighborsClassifier(),
        'params': {
            'model__n_neighbors': [3, 5, 7, 9],
            'model__weights': ['uniform', 'distance'],
            'model__p': [1, 2]
        }
    },
#     'svm': {
#         'model': SVC(),
#         'params': {
#             'model__C': [0.1, 1, 10],
#             'model__kernel': ['linear', 'rbf'],
#             'model__gamma': ['scale', 'auto']
#         }
#     }
}

# Create an empty list to store the best models
best_models = []

# Iterate through the pipeline dictionary and fit models using GridSearchCV
for model_name, config in pipeline_dict.items():
    print(f"\nRunning GridSearchCV for {model_name}")
    
    # Create a pipeline with a scaler (if needed) and the model
    if 'scaler' in config:
        pipeline = Pipeline([
            ('scaler', config['scaler']),
            ('model', config['model'])
        ])
    else:
        pipeline = Pipeline([
            ('model', config['model'])
        ])
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        config['params'],
        cv=5,  # 5-fold cross-validation
        scoring='f1_macro',
        n_jobs=-1  # Use all available CPUs
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and the corresponding model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    # Predict on the test set and calculate accuracy
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Best Parameters: {best_params}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    report = classification_report(y_test, y_pred, output_dict=True)
    f1_macro = report['macro avg']['f1-score']
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']

    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix for {model_name}:\n{cm}")
    
    # Save the best model
    best_models.append((model_name, best_model, accuracy, f1_macro, precision, recall))
    
    print("\nClassification Report:")
    print(classification_report(y_test, best_model.predict(X_test)))

best_models.sort(key=lambda x: (x[3]), reverse=True)
best_model_name, best_model, best_accuracy, best_f1_macro, best_precision, best_recall = best_models[0]

print(f"\nBest Model: {best_model_name}")
print(f"Best Test Accuracy: {best_accuracy:.4f}")
print(f"Best F1 Score (Macro): {best_f1_macro:.4f}")
print(f"Best Precision (Macro): {best_precision:.4f}")
print(f"Best Recall (Macro): {best_recall:.4f}")

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, best_model.predict(X_test)))


# Running the model after (Mean + Standard Deviation) Imputation without dropping any column

# In[103]:


X = df_nodrop_mpstd
Y = target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=17, stratify = Y) # stratify.


# In[104]:


# Create a pipeline dictionary with models and their hyperparameters
pipeline_dict = {
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'model__n_estimators': [25, 50, 100, 200],
            'model__max_depth': [10, 20, 30],
            'model__criterion':['gini', 'entropy', 'log_loss'],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2', None],
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(),
        'params': {
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l1', 'l2', 'elasticnet'],
            'model__max_iter': [100, 200, 300]
        }
    },
    
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'model__max_depth': [None, 5, 10, 20, 30],
            'model__min_samples_split': [1, 2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__criterion': ['gini', 'entropy'],
            'model__max_features': ['auto', 'sqrt', 'log2']
        }
    },
    'adaboost': {
        'model': AdaBoostClassifier(),
        'params': {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 1],
            'model__algorithm': ['SAMME', 'SAMME.R']
        }
    },
    'naive_bayes': {  
        'model': GaussianNB(),
        'params': {
            'model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
    },
    'knn': {
        'model': KNeighborsClassifier(),
        'params': {
            'model__n_neighbors': [3, 5, 7, 9],
            'model__weights': ['uniform', 'distance'],
            'model__p': [1, 2]
        }
    },
#     'svm': {
#         'model': SVC(),
#         'params': {
#             'model__C': [0.1, 1, 10],
#             'model__kernel': ['linear', 'rbf'],
#             'model__gamma': ['scale', 'auto']
#         }
#     }
}

# Create an empty list to store the best models
best_models = []

# Iterate through the pipeline dictionary and fit models using GridSearchCV
for model_name, config in pipeline_dict.items():
    print(f"\nRunning GridSearchCV for {model_name}")
    
    # Create a pipeline with a scaler (if needed) and the model
    if 'scaler' in config:
        pipeline = Pipeline([
            ('scaler', config['scaler']),
            ('model', config['model'])
        ])
    else:
        pipeline = Pipeline([
            ('model', config['model'])
        ])
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        config['params'],
        cv=5,  # 5-fold cross-validation
        scoring='f1_macro',
        n_jobs=-1  # Use all available CPUs
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and the corresponding model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    # Predict on the test set and calculate accuracy
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Best Parameters: {best_params}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    report = classification_report(y_test, y_pred, output_dict=True)
    f1_macro = report['macro avg']['f1-score']
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']

    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix for {model_name}:\n{cm}")
    
    # Save the best model
    best_models.append((model_name, best_model, accuracy, f1_macro, precision, recall))
    
    print("\nClassification Report:")
    print(classification_report(y_test, best_model.predict(X_test)))

best_models.sort(key=lambda x: (x[3]), reverse=True)
best_model_name, best_model, best_accuracy, best_f1_macro, best_precision, best_recall = best_models[0]

print(f"\nBest Model: {best_model_name}")
print(f"Best Test Accuracy: {best_accuracy:.4f}")
print(f"Best F1 Score (Macro): {best_f1_macro:.4f}")
print(f"Best Precision (Macro): {best_precision:.4f}")
print(f"Best Recall (Macro): {best_recall:.4f}")

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, best_model.predict(X_test)))


# Running the model after after (Mean + Standard Deviation) Imputation by dropping columns where more than 20% values are missing

# In[105]:


X = df_twenty_mpstd
Y = target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify = Y) 


# In[106]:


# Create a pipeline dictionary with models and their hyperparameters
pipeline_dict = {
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'model__n_estimators': [25, 50, 100, 200],
            'model__max_depth': [10, 20, 30],
            'model__criterion':['gini', 'entropy', 'log_loss'],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2', None],
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(),
        'params': {
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l1', 'l2', 'elasticnet'],
            'model__max_iter': [100, 200, 300]
        }
    },
    
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'model__max_depth': [None, 5, 10, 20, 30],
            'model__min_samples_split': [1, 2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__criterion': ['gini', 'entropy'],
            'model__max_features': ['auto', 'sqrt', 'log2']
        }
    },
    'adaboost': {
        'model': AdaBoostClassifier(),
        'params': {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 1],
            'model__algorithm': ['SAMME', 'SAMME.R']
        }
    },
    'naive_bayes': {  
        'model': GaussianNB(),
        'params': {
            'model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
    },
    'knn': {
        'model': KNeighborsClassifier(),
        'params': {
            'model__n_neighbors': [3, 5, 7, 9],
            'model__weights': ['uniform', 'distance'],
            'model__p': [1, 2]
        }
    },
#     'svm': {
#         'model': SVC(),
#         'params': {
#             'model__C': [0.1, 1, 10],
#             'model__kernel': ['linear', 'rbf'],
#             'model__gamma': ['scale', 'auto']
#         }
#     }
}

# Create an empty list to store the best models
best_models = []

# Iterate through the pipeline dictionary and fit models using GridSearchCV
for model_name, config in pipeline_dict.items():
    print(f"\nRunning GridSearchCV for {model_name}")
    
    # Create a pipeline with a scaler (if needed) and the model
    if 'scaler' in config:
        pipeline = Pipeline([
            ('scaler', config['scaler']),
            ('model', config['model'])
        ])
    else:
        pipeline = Pipeline([
            ('model', config['model'])
        ])
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        config['params'],
        cv=5,  # 5-fold cross-validation
        scoring='f1_macro',
        n_jobs=-1  # Use all available CPUs
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and the corresponding model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    # Predict on the test set and calculate accuracy
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Best Parameters: {best_params}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    report = classification_report(y_test, y_pred, output_dict=True)
    f1_macro = report['macro avg']['f1-score']
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']

    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix for {model_name}:\n{cm}")

    
    # Save the best model
    best_models.append((model_name, best_model, accuracy, f1_macro, precision, recall))
    
    print("\nClassification Report:")
    print(classification_report(y_test, best_model.predict(X_test)))

best_models.sort(key=lambda x: (x[3]), reverse=True)
best_model_name, best_model, best_accuracy, best_f1_macro, best_precision, best_recall = best_models[0]

print(f"\nBest Model: {best_model_name}")
print(f"Best Test Accuracy: {best_accuracy:.4f}")
print(f"Best F1 Score (Macro): {best_f1_macro:.4f}")
print(f"Best Precision (Macro): {best_precision:.4f}")
print(f"Best Recall (Macro): {best_recall:.4f}")

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, best_model.predict(X_test)))


# Running the model after (Mean - Standard Deviation) Imputation where no column is dropped

# In[107]:


X = df_nodrop_mmstd
Y = target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify = Y) 


# In[108]:


# pipeline dictionary with models and their hyperparameters
pipeline_dict = {
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'model__n_estimators': [25, 50, 100, 200],
            'model__max_depth': [10, 20, 30],
            'model__criterion':['gini', 'entropy', 'log_loss'],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2', None],
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(),
        'params': {
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l1', 'l2', 'elasticnet'],
            'model__max_iter': [100, 200, 300]
        }
    },
    
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'model__max_depth': [None, 5, 10, 20, 30],
            'model__min_samples_split': [1, 2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__criterion': ['gini', 'entropy'],
            'model__max_features': ['auto', 'sqrt', 'log2']
        }
    },
    'adaboost': {
        'model': AdaBoostClassifier(),
        'params': {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 1],
            'model__algorithm': ['SAMME', 'SAMME.R']
        }
    },
    'naive_bayes': {  
        'model': GaussianNB(),
        'params': {
            'model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
    },
    'knn': {
        'model': KNeighborsClassifier(),
        'params': {
            'model__n_neighbors': [3, 5, 7, 9],
            'model__weights': ['uniform', 'distance'],
            'model__p': [1, 2]
        }
    },
#     'svm': {
#         'model': SVC(),
#         'params': {
#             'model__C': [0.1, 1, 10],
#             'model__kernel': ['linear', 'rbf'],
#             'model__gamma': ['scale', 'auto']
#         }
#     }
}

# Create an empty list to store the best models
best_models = []

# Iterate through the pipeline dictionary and fit models using GridSearchCV
for model_name, config in pipeline_dict.items():
    print(f"\nRunning GridSearchCV for {model_name}")
    
    # Create a pipeline with a scaler (if needed) and the model
    if 'scaler' in config:
        pipeline = Pipeline([
            ('scaler', config['scaler']),
            ('model', config['model'])
        ])
    else:
        pipeline = Pipeline([
            ('model', config['model'])
        ])
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        config['params'],
        cv=5,  # 5-fold cross-validation
        scoring='f1_macro',
        n_jobs=-1  # Use all available CPUs
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and the corresponding model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    # Predict on the test set and calculate accuracy
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Best Parameters: {best_params}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    report = classification_report(y_test, y_pred, output_dict=True)
    f1_macro = report['macro avg']['f1-score']
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']

    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix for {model_name}:\n{cm}")
    
    # Save the best model
    best_models.append((model_name, best_model, accuracy, f1_macro, precision, recall))
    
    print("\nClassification Report:")
    print(classification_report(y_test, best_model.predict(X_test)))

best_models.sort(key=lambda x: (x[3]), reverse=True)
best_model_name, best_model, best_accuracy, best_f1_macro, best_precision, best_recall = best_models[0]

print(f"\nBest Model: {best_model_name}")
print(f"Best Test Accuracy: {best_accuracy:.4f}")
print(f"Best F1 Score (Macro): {best_f1_macro:.4f}")
print(f"Best Precision (Macro): {best_precision:.4f}")
print(f"Best Recall (Macro): {best_recall:.4f}")

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, best_model.predict(X_test)))


# Running the model after (Mean - Standard Deviation) Imputation by dropping columns where more than 20% values are missing 

# In[109]:


X = df_twenty_mmstd
Y = target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=17, stratify = Y) 


# In[110]:


# Create a pipeline dictionary with models and their hyperparameters
pipeline_dict = {
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'model__n_estimators': [25, 50, 100, 200],
            'model__max_depth': [10, 20, 30],
            'model__criterion':['gini', 'entropy', 'log_loss'],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2', None],
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(),
        'params': {
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l1', 'l2', 'elasticnet'],
            'model__max_iter': [100, 200, 300]
        }
    },
    
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'model__max_depth': [None, 5, 10, 20, 30],
            'model__min_samples_split': [1, 2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__criterion': ['gini', 'entropy'],
            'model__max_features': ['auto', 'sqrt', 'log2']
        }
    },
    'adaboost': {
        'model': AdaBoostClassifier(),
        'params': {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 1],
            'model__algorithm': ['SAMME', 'SAMME.R']
        }
    },
    'naive_bayes': {  
        'model': GaussianNB(),
        'params': {
            'model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
    },
    'knn': {
        'model': KNeighborsClassifier(),
        'params': {
            'model__n_neighbors': [3, 5, 7, 9],
            'model__weights': ['uniform', 'distance'],
            'model__p': [1, 2]
        }
    },
#     'svm': {
#         'model': SVC(),
#         'params': {
#             'model__C': [0.1, 1, 10],
#             'model__kernel': ['linear', 'rbf'],
#             'model__gamma': ['scale', 'auto']
#         }
#     }
}

# Create an empty list to store the best models
best_models = []

# Iterate through the pipeline dictionary and fit models using GridSearchCV
for model_name, config in pipeline_dict.items():
    print(f"\nRunning GridSearchCV for {model_name}")
    
    # Create a pipeline with a scaler (if needed) and the model
    if 'scaler' in config:
        pipeline = Pipeline([
            ('scaler', config['scaler']),
            ('model', config['model'])
        ])
    else:
        pipeline = Pipeline([
            ('model', config['model'])
        ])
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        config['params'],
        cv=5,  # 5-fold cross-validation
        scoring='f1_macro',
        n_jobs=-1  # Use all available CPUs
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and the corresponding model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    # Predict on the test set and calculate accuracy
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Best Parameters: {best_params}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    report = classification_report(y_test, y_pred, output_dict=True)
    f1_macro = report['macro avg']['f1-score']
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']

    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix for {model_name}:\n{cm}")
    
    # Save the best model
    best_models.append((model_name, best_model, accuracy, f1_macro, precision, recall))
    
    print("\nClassification Report:")
    print(classification_report(y_test, best_model.predict(X_test)))

best_models.sort(key=lambda x: (x[3]), reverse=True)
best_model_name, best_model, best_accuracy, best_f1_macro, best_precision, best_recall = best_models[0]

print(f"\nBest Model: {best_model_name}")
print(f"Best Test Accuracy: {best_accuracy:.4f}")
print(f"Best F1 Score (Macro): {best_f1_macro:.4f}")
print(f"Best Precision (Macro): {best_precision:.4f}")
print(f"Best Recall (Macro): {best_recall:.4f}")

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, best_model.predict(X_test)))


# Running the model after KNN Imputation by where no column is dropped

# In[111]:


X = df_nodrop_knn
Y = target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify = Y) 


# In[112]:


# Create a pipeline dictionary with models and their hyperparameters
pipeline_dict = {
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'model__n_estimators': [25, 50, 100, 200],
            'model__max_depth': [10, 20, 30],
            'model__criterion':['gini', 'entropy', 'log_loss'],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2', None],
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(),
        'params': {
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l1', 'l2', 'elasticnet'],
            'model__max_iter': [100, 200, 300]
        }
    },
    
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'model__max_depth': [None, 5, 10, 20, 30],
            'model__min_samples_split': [1, 2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__criterion': ['gini', 'entropy'],
            'model__max_features': ['auto', 'sqrt', 'log2']
        }
    },
    'adaboost': {
        'model': AdaBoostClassifier(),
        'params': {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 1],
            'model__algorithm': ['SAMME', 'SAMME.R']
        }
    },
    'naive_bayes': {  
        'model': GaussianNB(),
        'params': {
            'model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
    },
    'knn': {
        'model': KNeighborsClassifier(),
        'params': {
            'model__n_neighbors': [3, 5, 7, 9],
            'model__weights': ['uniform', 'distance'],
            'model__p': [1, 2]
        }
    },
#     'svm': {
#         'model': SVC(),
#         'params': {
#             'model__C': [0.1, 1, 10],
#             'model__kernel': ['linear', 'rbf'],
#             'model__gamma': ['scale', 'auto']
#         }
#     }
}

# Create an empty list to store the best models
best_models = []

# Iterate through the pipeline dictionary and fit models using GridSearchCV
for model_name, config in pipeline_dict.items():
    print(f"\nRunning GridSearchCV for {model_name}")
    
    # Create a pipeline with a scaler (if needed) and the model
    if 'scaler' in config:
        pipeline = Pipeline([
            ('scaler', config['scaler']),
            ('model', config['model'])
        ])
    else:
        pipeline = Pipeline([
            ('model', config['model'])
        ])
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        config['params'],
        cv=5,  # 5-fold cross-validation
        scoring='f1_macro',
        n_jobs=-1  # Use all available CPUs
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and the corresponding model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    # Predict on the test set and calculate accuracy
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Best Parameters: {best_params}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    report = classification_report(y_test, y_pred, output_dict=True)
    f1_macro = report['macro avg']['f1-score']
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']

    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix for {model_name}:\n{cm}")
    
    # Save the best model
    best_models.append((model_name, best_model, accuracy, f1_macro, precision, recall))
    
    print("\nClassification Report:")
    print(classification_report(y_test, best_model.predict(X_test)))

best_models.sort(key=lambda x: (x[3]), reverse=True)
best_model_name, best_model, best_accuracy, best_f1_macro, best_precision, best_recall = best_models[0]

print(f"\nBest Model: {best_model_name}")
print(f"Best Test Accuracy: {best_accuracy:.4f}")
print(f"Best F1 Score (Macro): {best_f1_macro:.4f}")
print(f"Best Precision (Macro): {best_precision:.4f}")
print(f"Best Recall (Macro): {best_recall:.4f}")

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, best_model.predict(X_test)))


# Running the model after KNN Imputation by dropping columns where more than 20% values are missing 


X = df_twenty_mmstd
Y = target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify = Y) 

# Create a pipeline dictionary with models and their hyperparameters
pipeline_dict = {
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'model__n_estimators': [25, 50, 100, 200],
            'model__max_depth': [10, 20, 30],
            'model__criterion':['gini', 'entropy', 'log_loss'],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2', None],
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(),
        'params': {
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l1', 'l2', 'elasticnet'],
            'model__max_iter': [100, 200, 300]
        }
    },
    
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'model__max_depth': [None, 5, 10, 20, 30],
            'model__min_samples_split': [1, 2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__criterion': ['gini', 'entropy'],
            'model__max_features': ['auto', 'sqrt', 'log2']
        }
    },
    'adaboost': {
        'model': AdaBoostClassifier(),
        'params': {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 1],
            'model__algorithm': ['SAMME', 'SAMME.R']
        }
    },
    'naive_bayes': {  
        'model': GaussianNB(),
        'params': {
            'model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
    },
    'knn': {
        'model': KNeighborsClassifier(),
        'params': {
            'model__n_neighbors': [3, 5, 7, 9],
            'model__weights': ['uniform', 'distance'],
            'model__p': [1, 2]
        }
    },
#     'svm': {
#         'model': SVC(),
#         'params': {
#             'model__C': [0.1, 1, 10],
#             'model__kernel': ['linear', 'rbf'],
#             'model__gamma': ['scale', 'auto']
#         }
#     }
}

# Create an empty list to store the best models
best_models = []

# Iterate through the pipeline dictionary and fit models using GridSearchCV
for model_name, config in pipeline_dict.items():
    print(f"\nRunning GridSearchCV for {model_name}")
    
    # Create a pipeline with a scaler (if needed) and the model
    if 'scaler' in config:
        pipeline = Pipeline([
            ('scaler', config['scaler']),
            ('model', config['model'])
        ])
    else:
        pipeline = Pipeline([
            ('model', config['model'])
        ])
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        config['params'],
        cv=5,  # 5-fold cross-validation
        scoring='f1_macro',
        n_jobs=-1  # Use all available CPUs
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and the corresponding model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    # Predict on the test set and calculate accuracy
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Best Parameters: {best_params}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    report = classification_report(y_test, y_pred, output_dict=True)
    f1_macro = report['macro avg']['f1-score']
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']

    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix for {model_name}:\n{cm}")
    
    # Save the best model
    best_models.append((model_name, best_model, accuracy, f1_macro, precision, recall))
    
    print("\nClassification Report:")
    print(classification_report(y_test, best_model.predict(X_test)))

best_models.sort(key=lambda x: (x[3]), reverse=True)
best_model_name, best_model, best_accuracy, best_f1_macro, best_precision, best_recall = best_models[0]

print(f"\nBest Model: {best_model_name}")
print(f"Best Test Accuracy: {best_accuracy:.4f}")
print(f"Best F1 Score (Macro): {best_f1_macro:.4f}")
print(f"Best Precision (Macro): {best_precision:.4f}")
print(f"Best Recall (Macro): {best_recall:.4f}")

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, best_model.predict(X_test)))


# ## Running the Model on the training data

df_test = pd.read_csv('test_data (1).csv')
df_test_encoded = pd.get_dummies(df, columns=['hypertensive', 'atrialfibrillation', 'CHD with no MI', 'diabetes', 'deficiencyanemias', 'depression', 'Hyperlipemia', 'Renal failure', 'COPD'], prefix=('hypertensive', 'atrialfibrillation', 'CHD with no MI', 'diabetes', 'deficiencyanemias', 'depression', 'Hyperlipemia', 'Renal failure', 'COPD'))                          
print(df_encoded)

for column in df_test_encoded.columns:
    median = df_test_encoded[column].median()
    df_test_encoded[column].fillna(median, inplace = True)

X_train_final = df_nodrop_median.copy()
y_train = target.copy()

adaboost_model = AdaBoostClassifier(
    algorithm='SAMME',
    learning_rate=1,
    n_estimators=100,
    random_state=42
)

adaboost_model.fit(X_train_final, y_train)


y_pred = adaboost_model.predict(df_test_encoded)
y_pred


feature_importances = adaboost_model.feature_importances_

for feature, importance in zip(X_train_final.columns, feature_importances):
    print(f"{feature}: {importance}")

save_file = np.savetxt("Rohan_Mehra_21224_ML_Labels_final.txt", y_pred, delimiter="\n", fmt="%.0f")


