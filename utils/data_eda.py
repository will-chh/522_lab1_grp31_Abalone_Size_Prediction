import altair as alt
import pandas as pd
import click

NEW_COLUMN_NAMES = [
    "Length", "Diameter", "Height",
    "Whole_weight", "Shucked_weight",
    "Viscera_weight", "Shell_weight", "Rings"
]

def scatter_matrix(df: pd.DataFrame) -> alt.Chart:
    chart = (
        alt.Chart(df, width=150, height=100)
        .mark_point()
        .encode(
            alt.X(alt.repeat("row"), type="quantitative"),
            alt.Y(alt.repeat("column"), type="quantitative"),
            color=alt.Color("Sex:N", title="Abalone Sex"),
        )
        .repeat(column=NEW_COLUMN_NAMES, row=NEW_COLUMN_NAMES)
        .properties(title="Scatterplot matrix of abalone physical features and rings")
    )
    return chart

@click.command()
@click.option("--input_path", required=True, type=str, help="Path to validated CSV")
@click.option("--output_path", required=True, type=str, help="Path to save the scatter matrix figure")
def main(input_path, output_path):
    df = pd.read_csv(input_path)
    chart = scatter_matrix(df)
    chart.save(output_path)
    print(f"Scatter matrix saved to {output_path}")

if __name__ == "__main__":
    main()