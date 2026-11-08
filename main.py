import polars as pl


def main():
    data = pl.read_csv("Lugeja25.tab", has_header=True, separator="\t")
    print(data)

if __name__ == "__main__":
    main()
