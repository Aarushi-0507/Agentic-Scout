"""
Unit tests for DataLoader and data ingestion.
"""
import os
import pytest
import pandas as pd
from core.loader import DataLoader


@pytest.fixture
def loader():
    return DataLoader()


def test_load_csv_file(loader, tmp_path):
    df_test = pd.DataFrame({
        "A": [1, 2, 3, 4, 5, 6],
        "B": [10.5, 20.2, 30.1, 40.8, 50.0, 60.5],
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"]
    })
    csv_file = tmp_path / "test.csv"
    df_test.to_csv(csv_file, index=False)
    
    loaded_df, meta = loader.load(str(csv_file))
    assert loaded_df.shape == (6, 3)
    assert meta["rows"] == 6
    assert meta["columns"] == 3
    assert pd.api.types.is_datetime64_any_dtype(loaded_df["Date"])


def test_load_excel_file(loader, tmp_path):
    df_test = pd.DataFrame({
        "Category": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
        "Value": [100, 200, 300, 400, 500]
    })
    xlsx_file = tmp_path / "test.xlsx"
    df_test.to_excel(xlsx_file, index=False)
    
    loaded_df, meta = loader.load(str(xlsx_file))
    assert loaded_df.shape == (5, 2)
    assert meta["rows"] == 5


def test_load_nonexistent_file(loader):
    with pytest.raises(FileNotFoundError):
        loader.load("non_existent_file.csv")


def test_load_empty_file(loader, tmp_path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")
    with pytest.raises(ValueError, match="empty"):
        loader.load(str(empty_file))


def test_load_insufficient_columns(loader, tmp_path):
    single_col = tmp_path / "single.csv"
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
    df.to_csv(single_col, index=False)
    with pytest.raises(ValueError, match="at least 2 columns"):
        loader.load(str(single_col))
