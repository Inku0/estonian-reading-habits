import numpy as np
import typol as tp
from catboost import CatBoostClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from Answer import (
    AnswerParts1_6_NoQ1_V445_Q3a15_Q3a14,
    AnswerParts1_6WithQ1_V445_Q3a15_Q3a14,
    BinaryYesOrNo,
    FeatureImportance,
    PredictedAndActualQ1,
)


def binary_is_reader(x: int) -> int:
    if x == 1:
        return 0
    else:
        return 1


def split_data_for_parts_1_6(
    data: tp.DataFrame[AnswerParts1_6WithQ1_V445_Q3a15_Q3a14],
) -> tuple[
    tp.DataFrame[AnswerParts1_6_NoQ1_V445_Q3a15_Q3a14],
    tp.DataFrame[AnswerParts1_6_NoQ1_V445_Q3a15_Q3a14],
    tp.Series[BinaryYesOrNo],
    tp.Series[BinaryYesOrNo],
]:
    data_without_Q1_V445 = data.transform(AnswerParts1_6_NoQ1_V445_Q3a15_Q3a14)

    data_with_binary_Q1 = data.with_columns(data.s.Q1.map(binary_is_reader))
    only_binary_Q1 = data_with_binary_Q1[data_with_binary_Q1.s.Q1]

    X_train, X_test, y_train, y_test = train_test_split(
        data_without_Q1_V445.dataframe,
        only_binary_Q1.to_list(),
        test_size=0.20,
        random_state=0,
    )

    X_train = tp.DataFrame(AnswerParts1_6_NoQ1_V445_Q3a15_Q3a14, X_train)
    X_test = tp.DataFrame(AnswerParts1_6_NoQ1_V445_Q3a15_Q3a14, X_test)
    y_train: tp.Series[BinaryYesOrNo] = tp.Series(y_train)
    y_test: tp.Series[BinaryYesOrNo] = tp.Series(y_test)

    return (X_train, X_test, y_train, y_test)

def train_predict_is_reader_parts_1_6(X_train: tp.DataFrame[AnswerParts1_6_NoQ1_V445_Q3a15_Q3a14], y_train: tp.Series[BinaryYesOrNo]) -> CatBoostClassifier:
    # we're trying to predict the enum HowManyBooksReadInTheLastSixMonths aka Question 11

    # we need to predict whether a person considers themself a reader.
    # we can only predict this based on the answers they give in parts 1 and 6,
    # because they're the only parts that non-readers and readers both fill in
    is_reader_predictor = CatBoostClassifier(
        verbose=False,
        random_state=0,
        border_count=32,
        random_strength=0.45,
        depth=6,
        auto_class_weights="Balanced",
        learning_rate=0.1,
        l2_leaf_reg=10,
        iterations=500,
    )
    is_reader_predictor.fit(X_train.dataframe, y_train.data)
    return is_reader_predictor

def predict_is_reader_parts_1_6(is_reader_predictor: CatBoostClassifier, X_test: tp.DataFrame[AnswerParts1_6_NoQ1_V445_Q3a15_Q3a14], y_test: tp.Series[BinaryYesOrNo]) -> tp.DataFrame[PredictedAndActualQ1]:
    predictions: np.ndarray[tuple[int], np.dtype[np.int32]] = is_reader_predictor.predict(X_test.dataframe)
    as_dict = {
        "Predicted": predictions,
        "Actual": y_test.data,
    }
    results = tp.DataFrame(PredictedAndActualQ1, as_dict)  # ty: ignore[no-matching-overload]
    return results  # ty: ignore[invalid-return-type]

def visualize_is_reader_predictions(data: tp.DataFrame[PredictedAndActualQ1]):
    actual = data[data.s.Actual].to_list()
    predicted = data[data.s.Predicted].to_list()

    cm = confusion_matrix(actual, predicted, labels=[1, 0])
    correct = np.eye(2, dtype=bool)

    fig, ax = plt.subplots()

    greatest_error = max(int(cm[~correct].max()), 15)
    ax.imshow(np.ma.masked_where(~correct, cm), vmin=0, vmax=greatest_error, cmap="Greens", origin="lower")
    ax.imshow(np.ma.masked_where(correct, cm), vmin=0, vmax=greatest_error, cmap="Reds", origin="lower")

    for row in range(2):
        for column in range(2):
            vmax = cm[correct].max() if correct[row, column] else cm[~correct].max()
            ax.text(
                column, row, cm[row, column],
                ha="center",
                va="center",
                weight="bold",
                color="white" if (cm[row, column] / vmax) > 0.01 else "black",
            )

    ax.set_xticks([0, 1], ["Reader", "Non-reader"])
    ax.set_yticks([0, 1], ["Reader", "Non-reader"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Do you read books?")
    fig.tight_layout()
    plt.show()

def find_best_params_for_catboost(data: tp.DataFrame[AnswerParts1_6WithQ1_V445_Q3a15_Q3a14]) -> dict[str, dict[str, str]]:
    grid = {
        "learning_rate": [0.01, 0.1],
        "depth": [5, 6, 7],
        "l2_leaf_reg": [9, 10, 11],
        "iterations": [1000],
        "border_count": [16, 32, 64],
        "random_strength": [0.45, 0.5, 0.55],
        "bagging_temperature": [0, 0.1, 0.5],
        "auto_class_weights": ["None", "Balanced"]
    }
    is_reader_predictor = CatBoostClassifier(
        verbose=True,
        random_state=0,
    )
    search_results: dict[str, dict[str, str]] = is_reader_predictor.grid_search(
        X=data.transform(AnswerParts1_6_NoQ1_V445_Q3a15_Q3a14).dataframe,
        y=data.with_columns(
            data.s.Q1.map(binary_is_reader)
        )[data.s.Q1].to_list(),
        verbose=True,
        partition_random_seed=0,
        param_grid=grid
    )
    print(search_results)
    return search_results

def is_reader_PCA(is_reader_predictor: CatBoostClassifier) -> tp.DataFrame[FeatureImportance]:
    return is_reader_predictor.get_feature_importance(prettified=True)