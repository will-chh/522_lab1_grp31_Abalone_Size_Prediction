import pandas as pd
from pandera import Column, Check, DataFrameSchema
import click

# Allowed categories for Sex
SEX_CATEGORIES = ["M", "F", "I"]

abalone_schema = DataFrameSchema(
    {
        "Sex": Column(
            str,
            Check.isin(SEX_CATEGORIES),
            nullable=False
        ),
        "Length": Column(
            float,
            Check.ge(0.0),
            nullable=False
        ),
        "Diameter": Column(
            float,
            Check.ge(0.0),
            nullable=False
        ),
        "Height": Column(
            float,
            Check.ge(0.0),
            nullable=False
        ),
        "Whole_weight": Column(
            float,
            Check.ge(0.0),
            nullable=False
        ),
        "Shucked_weight": Column(
            float,
            Check.ge(0.0),
            nullable=False
        ),
        "Viscera_weight": Column(
            float,
            Check.ge(0.0),
            nullable=False
        ),
        "Shell_weight": Column(
            float,
            Check.ge(0.0),
            nullable=False
        ),
        "Rings": Column(
            int,
            Check.between(1, 30),
            nullable=False
        )
    },
    checks=[
        Check(lambda df: ~df.duplicated().any(), error="Duplicate rows found."),
        Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found."),
        Check(lambda df: (df.isna().mean() <= 0.05).all(),
              error="Missingness exceeds 5% threshold.")
    ]
)

column_names = [
    "Sex", "Length", "Diameter", "Height",
    "Whole_weight", "Shucked_weight",
    "Viscera_weight", "Shell_weight", "Rings"
]

def load_and_validate_abalone(url) -> pd.DataFrame:
    #1) Loading data 
    abalone_raw = pd.read_csv(url, header=None, names=column_names)

    #2) validation with pandera
    abalone_validated = abalone_schema.validate(abalone_raw, lazy=True)

    return abalone_validated

@click.command()
@click.option("--input_path", required=True, type=str, help="URL or local path to raw CSV")
@click.option("--output_path", required=True, type=str, help="Path to save validated CSV")
def main(input_path, output_path):
    abalone_validated = load_and_validate_abalone(input_path)
    abalone_validated.to_csv(output_path, index=False)
    print(f"Validated data saved to {output_path}")

if __name__ == "__main__":
    main()