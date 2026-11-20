import polars as pl
import typol as tp

from Answer import Answer


def main():
    raw_data: pl.DataFrame = pl.read_csv("Lugeja25.tab", has_header=True, separator="\t")
    typed_data: tp.DataFrame[Answer] = tp.DataFrame(Answer, raw_data)
    print(typed_data.head(1))

if __name__ == "__main__":
    main()
