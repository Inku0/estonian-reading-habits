import numpy as np
import typol as tp
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from Answer import AnswerParts1_2_3_4_5_6_NotQ11, CompleteAnswer, PredictedAndActualQ11

labels = {
    0: "None",
    1: "1...2",
    2: "3...4",
    3: "5...6",
    4: "6...10",
    5: "10...20",
    6: ">20",
}


def predict_habits(data: tp.DataFrame[CompleteAnswer]) -> tp.DataFrame[PredictedAndActualQ11]:
    # we're trying to predict the enum HowManyBooksReadInTheLastSixMonths aka Question 11
    data = data.filter(data.s.Q1 > 1)
    data = data.with_columns(data.s.Q11_1.fill_null(66)).filter(data.s.Q11_1 != 66)  # ty: ignore[no-matching-overload, invalid-argument-type]
    training_data = data.transform(AnswerParts1_2_3_4_5_6_NotQ11)  # ty: ignore[unresolved-attribute]
    X_train, X_test, y_train, y_test = train_test_split(
        training_data.dataframe,
        data[CompleteAnswer.Q11_1].to_list(),
        test_size=0.30,
        random_state=0
    )
    books_read_predictor = RandomForestClassifier()
    books_read_predictor.fit(X_train, y_train)
    predictions: np.ndarray[tuple[int], np.dtype[np.int32]] = books_read_predictor.predict(X_test)
    as_dict = {
        "Predicted": predictions,
        "Actual": y_test,
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

    ax.imshow(np.ma.masked_where(~correct, cm), cmap="Greens", origin="lower")
    ax.imshow(np.ma.masked_where(correct, cm), cmap="Reds", origin="lower")

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