import polars as pl
import typol as tp

from Answer import Answer


# TODO Predict a person's reading habits (how much/often they read) according to their answers
# TODO Assess the factors most responsible for determining a person's reading habits (PCA)
# TODO Determine the feasibility of a model for recommending books to a person according to their answers
def main():
    raw_data: pl.DataFrame = pl.read_csv("Lugeja25.tab", has_header=True, separator="\t")
    typed_data: tp.DataFrame[Answer] = tp.DataFrame(Answer, raw_data)
    print(typed_data.head(1))

if __name__ == "__main__":
    main()
