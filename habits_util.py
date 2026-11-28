import numpy as np
import typol as tp
from catboost import CatBoostClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from Answer import (
    AnswerParts1_2_3_4_5_6_NotQ11,
    BinaryYesOrNo,
    CompleteAnswer,
    FeatureImportance,
    PredictedAndActualQ11, HowManyBooksReadInTheLastSixMonths,
)

labels = {
    0: "None",
    1: "1...2",
    2: "3...4",
    3: "5...6",
    4: "6...10",
    5: "10...20",
    6: ">20",
}


def split_data_for_habits(
    data: tp.DataFrame[CompleteAnswer],
) -> tuple[
    tp.DataFrame[AnswerParts1_2_3_4_5_6_NotQ11],
    tp.DataFrame[AnswerParts1_2_3_4_5_6_NotQ11],
    tp.Series[HowManyBooksReadInTheLastSixMonths],
    tp.Series[HowManyBooksReadInTheLastSixMonths],
]:
    data = data.filter(data.s.Q1 > 1)
    data = data.with_columns(data.s.Q11_1.fill_null(66)).filter(data.s.Q11_1 != 66)  # ty: ignore[no-matching-overload, invalid-argument-type]

    data_without_Q11 = data.transform(AnswerParts1_2_3_4_5_6_NotQ11)

    X_train, X_test, y_train, y_test = train_test_split(
        data_without_Q11.dataframe,
        data[data.s.Q11_1].to_list(),
        test_size=0.20,
        random_state=0,
    )

    X_train = tp.DataFrame(AnswerParts1_2_3_4_5_6_NotQ11, X_train)
    X_test = tp.DataFrame(AnswerParts1_2_3_4_5_6_NotQ11, X_test)
    y_train: tp.Series[HowManyBooksReadInTheLastSixMonths] = tp.Series(y_train)
    y_test: tp.Series[HowManyBooksReadInTheLastSixMonths] = tp.Series(y_test)

    return (X_train, X_test, y_train, y_test)

def train_predict_habits(X_train: tp.DataFrame[AnswerParts1_2_3_4_5_6_NotQ11], y_train: tp.Series[HowManyBooksReadInTheLastSixMonths]) -> CatBoostClassifier:
    # we're trying to predict the enum HowManyBooksReadInTheLastSixMonths aka Question 11
    habits_predictor = CatBoostClassifier(
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
    habits_predictor.fit(X_train.dataframe, y_train.data)
    return habits_predictor

def predict_habits(habits_predictor: CatBoostClassifier, X_test: tp.DataFrame[AnswerParts1_2_3_4_5_6_NotQ11], y_test: tp.Series[HowManyBooksReadInTheLastSixMonths]) -> tp.DataFrame[PredictedAndActualQ11]:
    predictions: np.ndarray[tuple[int], np.dtype[np.int32]] = habits_predictor.predict(X_test.dataframe)
    arr = predictions.reshape(-1) # converts from a matrix to an array
    as_dict = {
        "Predicted": arr,
        "Actual": y_test.data,
    }
    results = tp.DataFrame(PredictedAndActualQ11, as_dict)  # ty: ignore[no-matching-overload]
    return results  # ty: ignore[invalid-return-type]

def visualize_predict_habits(data: tp.DataFrame[PredictedAndActualQ11]):
    actual = data[data.s.Actual].to_list()
    predicted = data[data.s.Predicted].to_list()

    cm = confusion_matrix(
        actual,
        predicted,
        # labels=list(labels.values())
    )
    correct = np.eye(7, dtype=bool)

    fig, ax = plt.subplots()

    greatest_error = max(int(cm[~correct].max()), 15)
    ax.imshow(np.ma.masked_where(~correct, cm), vmin=0, vmax=greatest_error, cmap="Greens", origin="lower")
    ax.imshow(np.ma.masked_where(correct, cm), vmin=0, vmax=greatest_error, cmap="Reds", origin="lower")

    for row in range(7):
        for column in range(7):
            vmax = cm[correct].max() if correct[row, column] else cm[~correct].max()
            ax.text(
                column, row, cm[row, column],
                ha="center",
                va="center",
                weight="bold",
                color="white" if (cm[row, column] / vmax) > 0.5 else "black",
            )

    ax.set_xticks(list(labels), list(labels.values()))
    ax.set_yticks(list(labels), list(labels.values()))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("How many books have you read in the last six months?")
    fig.tight_layout()
    plt.show()
    
def habits_PCA(habits_predictor: CatBoostClassifier) -> tp.DataFrame[FeatureImportance]:
    return habits_predictor.get_feature_importance(prettified=True)