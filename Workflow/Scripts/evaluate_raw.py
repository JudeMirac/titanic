"""Evaluate raw rows with training-only preprocessing; never overwrite artifacts."""
import argparse
import json
from pathlib import Path

FEATURES = ['Pclass', 'Age', 'SibSp', 'Sex', 'Cabin']


def build_pipeline():
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

    numeric = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ])
    categorical = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(handle_unknown='ignore', drop='if_binary')),
    ])
    preprocessing = ColumnTransformer([
        ('numeric', numeric, ['Pclass', 'Age', 'SibSp']),
        ('sex', categorical, ['Sex']),
        # fit_transform cross-fits on training rows; transform never sees test labels.
        ('cabin', TargetEncoder(target_type='binary', cv=5,
                                shuffle=True, random_state=42), ['Cabin']),
    ])
    return Pipeline([
        ('preprocessing', preprocessing),
        ('model', LogisticRegression(C=1, penalty='l1', class_weight='balanced',
                                     solver='liblinear', random_state=42)),
    ])


def evaluate(frame, threshold=0.45):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    missing = set(FEATURES + ['Survived']) - set(frame.columns)
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(sorted(missing)))
    if not 0 < threshold < 1:
        raise ValueError('threshold must be between zero and one')
    X_train, X_test, y_train, y_test = train_test_split(
        frame[FEATURES], frame['Survived'], test_size=0.3,
        random_state=42, stratify=frame['Survived'],
    )
    if y_train.value_counts().min() < 5 or y_train.nunique() != 2:
        raise ValueError('Training data needs at least five rows per class for cabin cross-fitting')
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        'training_rows': len(X_train), 'test_rows': len(X_test),
        'threshold': threshold,
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, zero_division=0),
        'recall': recall_score(y_test, predictions, zero_division=0),
        'f1': f1_score(y_test, predictions, zero_division=0),
        'roc_auc': roc_auc_score(y_test, probabilities),
        'scope': 'Raw-row evaluation with training-only preprocessing; no artifact saved.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'Data' / 'raw_data.csv')
    parser.add_argument('--threshold', type=float, default=0.45,
                        help='Fixed threshold; do not choose it from this run\'s test results.')
    args = parser.parse_args()
    import pandas as pd
    print(json.dumps(evaluate(pd.read_csv(args.data), args.threshold), indent=2))


if __name__ == '__main__':
    main()
