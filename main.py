import numpy as np
import polars as pl
import typol as tp
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from Answer import Answer, AnswerWithoutQ11


# TODO Predict a person's reading habits (how much/often they read) according to their answers
def predict_habits(data: tp.DataFrame[Answer]):
    # we're trying to predict the enum HowManyBooksReadInTheLastSixMonths aka Question 11
    training_data = data.transform(AnswerWithoutQ11)
    X_train, X_test, y_train, y_test = train_test_split(
        training_data, data.s.Q11_1, test_size=0.20, random_state=0
    )
    predictor = RandomForestRegressor()
    predictor.fit(X_train, y_train)
    results: np.ndarray = predictor.predict(X_test)
    print(results)
    
# TODO Assess the factors most responsible for determining a person's reading habits (PCA)
# TODO Determine the feasibility of a model for recommending books to a person according to their answers
    
def main():
    raw_data: pl.DataFrame = pl.read_csv("Lugeja25.tab", has_header=True, separator="\t")
    typed_data: tp.DataFrame[Answer] = tp.DataFrame(Answer, raw_data)
    print(typed_data.head(1))
    predict_habits(typed_data)

if __name__ == "__main__":
    main()
