"""Usage: data_analysis.py [TRAINDATA] [TESTDATA] [OUTPUTFOLDER] ...

Arguments:
  TRAINDATA        path of the training data
  TESTDATA         path of the testing data
  OUTPUTFOLDER     folder that stores the generated plots


"""
from docopt import docopt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder


def perform_ml_analysis(X_train_enc, X_test_enc, Y_train_data_path, Y_test_data_path):
    if(not os.path.exists(out_folder)):
        os.mkdir(out_folder)
        
    headernames = [
        "ID",
        "LIMIT_BAL",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
        "default payment next month",
    ]
        
    train_df = pd.read_csv(train_data_path, index_col=0, names=headernames, skiprows=1, encoding="utf-8")
    test_df = pd.read_csv(test_data_path, index_col=0, names=headernames, skiprows=1, encoding="utf-8")
        
    
    # 1. Create the column transformer / preprocessor
    # For the numeric features, we will Standardize the numbers. 
    # For the two categorical features, we will apply OneHotEncoder on them to convert
    # the column into multiple binary features. 
    # Many other features already have good encoding, so we don't
    # have to modify them (passthrough). 

    numeric_features = [
        "LIMIT_BAL",
        "AGE",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]

    categorical_features = ["MARRIAGE", "SEX"]

    drop = ["ID"]

    passthrough_features = [
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "EDUCATION",
    ]

    # Create the column transformer
    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (
            "passthrough",
            passthrough_features,
        ),
        (
            OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse=False),
            categorical_features,
        ),
        ("drop", drop),
    )

    # Show the preprocessor
    preprocessor
    
    # 2. Fit and transform on the training data
    X_train = train_df.drop(columns=["default payment next month"])
    X_test = test_df.drop(columns=["default payment next month"])
    y_train = train_df["default payment next month"]
    y_test = test_df["default payment next month"]

    # This line nicely formats the feature names from `preprocessor.get_feature_names_out()`
    # so that we can more easily use them below
    preprocessor.verbose_feature_names_out = False
    # Create a dataframe with the transformed features and column names
    preprocessor.fit(X_train)

    # transformed data
    X_train_transformed = preprocessor.transform(X_train)
    ohe_features = (
        preprocessor.named_transformers_["onehotencoder"].get_feature_names_out().tolist()
    )

    # Code to get all the feature names
    feature_names = numeric_features + passthrough_features + ohe_features

    X_train_enc = pd.DataFrame(X_train_transformed, columns=feature_names)

    # Show the transformed data
    X_train_enc
    
    import sys
    from hashlib import sha1

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import (
        GridSearchCV,
        RandomizedSearchCV,
        cross_validate,
        train_test_split,
    )
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import BernoulliNB, MultinomialNB
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import RidgeClassifier
    from sklearn.ensemble import RandomForestClassifier
    
    models = {
        # "dummy": DummyClassifier(random_state=123),
        "Decision Tree": DecisionTreeClassifier(random_state=123),
        "KNN": KNeighborsClassifier(),
        "RBF SVM": SVC(random_state=123),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=123),
        "Ridge_cla": RidgeClassifier(random_state=123),
        "RandomForest_cla": RandomForestClassifier(random_state=123),
    }

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    classification_metrics = ["accuracy", "precision", "recall", "f1"]

    from collections import defaultdict

    cross_val_results = defaultdict(list)
    for model in models:
        cross_val_results[model].append(
            cross_validate(
                make_pipeline(preprocessor, models[model]),
                X_train,
                y_train,
                cv=5,
                return_train_score=True,
                scoring=classification_metrics,
            )
        )
        
    # code below modified from :https://stackoverflow.com/questions/13575090/
    #              construct-pandas-dataframe-from-items-in-nested-dictionary
    cross_val_results_df = (
        pd.concat(
            {
                key: pd.DataFrame(value[0]).agg(["mean", "std"])
                for key, value in cross_val_results.items()
            },
            axis=0,
        )
        .T.style.format(
            precision=2  # Pandas `.style` does not honor previous rounding via `.round()`
        )
        .background_gradient(
            axis=None,
            vmax=1,
            vmin=0,  # Color cells based on the entire matrix rather than row/column-wise
        )
    )

    cross_val_results_df
    
    # Selecting RandomForestClassifier for model hyperparameter optimization. 
    from scipy.stats import expon, lognorm, loguniform, randint, uniform, norm
    from sklearn.model_selection import RandomizedSearchCV

    rf_cla = make_pipeline(preprocessor, RandomForestClassifier())
    # rf_cla.fit(X_train, y_train)

    param_dist = {"randomforestclassifier__max_depth": randint(1, 500),
                 "randomforestclassifier__class_weight": ["balanced", None]}

    random_search = RandomizedSearchCV(
        rf_cla,
        param_dist,
        n_iter=20,
        verbose=1,
        n_jobs=-1,
        random_state=123,
        scoring="f1",
    )

    random_search.fit(X_train, y_train)

    random_search.best_estimator_
    
    # This is an improvement compared to the mean validation f1 score using default parameters.
    random_search.best_score_
    
    # This is the optimized hyperparameter.
    random_search.best_params_
    
    # Generate a table showing feature importances.
    feat_importance = random_search.best_estimator_.named_steps[
        "randomforestclassifier"
    ].feature_importances_

    ohe_names = (
        random_search.best_estimator_.named_steps["columntransformer"]
        .named_transformers_["onehotencoder"]
        .get_feature_names_out()
        .tolist()
    )

    feat_names = numeric_features + ohe_names + passthrough_features

    pd.DataFrame({"Feature": feat_names, "Importance": feat_importance}).sort_values(
        by="Importance", ascending=False
    )
    
    
    # Add confusion matrix, predict score on test data.
    from sklearn.metrics import ConfusionMatrixDisplay

    cm = ConfusionMatrixDisplay.from_estimator(
        random_search.best_estimator_, X_test, y_test, values_format="d", display_labels=["Non default", "default"]
    )
    
    # Generate the classification report
    from sklearn.metrics import classification_report
    y_pred = random_search.best_estimator_.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Non default", "default"]))
    
    # Evaluate the Precision-Recall curve of the optimized model. 

    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(
        y_test, random_search.best_estimator_.predict_proba(X_test)[:, 1]
    )
    plt.plot(precision, recall, label="logistic regression: PR curve")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.plot(
        precision_score(y_test, random_search.best_estimator_.predict(X_test)),
        recall_score(y_test, random_search.best_estimator_.predict(X_test)),
        "or",
        markersize=10,
        label="threshold 0.5",
    )
    plt.legend(loc="best");
    plt.savefig('../results/precision_recall.png')
    
    # Evaluate the Receiver Operating Characteristic (ROC) curve of the optimized model. 
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_test, random_search.best_estimator_.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR (recall)")

    default_threshold = np.argmin(np.abs(thresholds - 0.5))

    plt.plot(
        fpr[default_threshold],
        tpr[default_threshold],
        "or",
        markersize=10,
        label="threshold 0.5",
    )
    plt.legend(loc="best");
    plt.savefig('../results/roc.png')
    
    # Finally, check the f1_score of the test data with our optimized model. 
    test_f1_score = f1_score(y_test, random_search.best_estimator_.predict(X_test))
    print(f"The f1_score of the test data is",round(test_f1_score, 3))
    
    
    
    
    
    
    
    
    
    

# Make sure you call this script in the utils folder
# Example: python3 data_eda.py ../data/processed/train_raw.csv ../data/eda_results/
if __name__ == '__main__':
    arguments = docopt(__doc__)

    train_data = arguments['TRAINDATA'] # Download 1 dataset at a time
    test_data = arguments['TESTDATA'] # Download 1 dataset at a time
    out_path = arguments['OUTPUTFOLDER'][0]
    ext_name = train_data_path.split('.')[-1]
    perform_ml_analysis(train_data, test_data, out_path)
