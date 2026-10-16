"""
Data loading and ingestion layer for AGENTIC SCOUT.
Supports CSV, Excel, with extensible architecture for SQL and APIs.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Union, BinaryIO, TextIO
import os
import io
import time
import pandas as pd


class BaseDataSource(ABC):
    """Abstract base class for all data ingestion sources."""
    
    @abstractmethod
    def load(self, source: Any, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load data into a pandas DataFrame along with metadata."""
        pass


class DataLoader(BaseDataSource):
    """
    Standard data loader for file-based datasets (CSV, Excel).
    Handles encoding fallbacks, date parsing, and validation.
    """
    
    def load(self, source: Union[str, BinaryIO, io.BytesIO], **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        start_time = time.time()
        file_name = kwargs.get("filename", "")
        file_size_bytes = 0
        
        if isinstance(source, str):
            if not os.path.exists(source):
                raise FileNotFoundError(f"File not found: {source}")
            file_name = os.path.basename(source)
            file_size_bytes = os.path.getsize(source)
            ext = os.path.splitext(file_name)[1].lower()
            
            if ext in [".csv", ".txt"]:
                df = self._load_csv(source)
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(source)
            else:
                raise ValueError(f"Unsupported file format: {ext}. Supported formats are CSV and Excel (.xlsx, .xls).")
        else:
            # Streamlit UploadedFile or BytesIO
            if not file_name and hasattr(source, "name"):
                file_name = source.name
            ext = os.path.splitext(file_name)[1].lower() if file_name else ".csv"
            
            if hasattr(source, "size"):
                file_size_bytes = source.size
            elif hasattr(source, "getbuffer"):
                file_size_bytes = source.getbuffer().nbytes
                
            if ext in [".csv", ".txt"]:
                df = self._load_csv_buffer(source)
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(source)
            else:
                raise ValueError(f"Unsupported file format: {ext}. Supported formats are CSV and Excel (.xlsx, .xls).")
                
        # Basic validation
        if df.empty:
            raise ValueError("The provided dataset is empty (0 rows).")
        if df.shape[1] < 2:
            raise ValueError("The dataset must contain at least 2 columns for meaningful analysis.")
        if df.shape[0] < 5:
            raise ValueError("The dataset must contain at least 5 rows for machine learning profiling.")

        # Compute content-based data signature
        import hashlib
        sig_hash = hashlib.md5()
        if isinstance(source, str):
            with open(source, "rb") as f:
                sig_hash.update(f.read())
        elif hasattr(source, "getvalue"):
            sig_hash.update(source.getvalue())
        elif hasattr(source, "getbuffer"):
            sig_hash.update(source.getbuffer())
        else:
            sig_hash.update(str(df.shape).encode())
            sig_hash.update(str(df.columns.tolist()).encode())
        data_signature = sig_hash.hexdigest()

        # Attempt safe date column parsing
        df = self._parse_potential_dates(df)

        load_duration_ms = (time.time() - start_time) * 1000.0
        metadata = {
            "filename": file_name,
            "data_signature": data_signature,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "size_bytes": file_size_bytes,
            "size_kb": round(file_size_bytes / 1024.0, 2),
            "load_duration_ms": round(load_duration_ms, 2)
        }
        
        return df, metadata

    def _load_csv(self, filepath: str) -> pd.DataFrame:
        encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]
        for enc in encodings:
            try:
                return pd.read_csv(filepath, encoding=enc)
            except pd.errors.EmptyDataError:
                raise ValueError("The provided dataset is empty (0 rows).")
            except UnicodeDecodeError:
                continue
        try:
            return pd.read_csv(filepath, encoding="utf-8", errors="replace")
        except pd.errors.EmptyDataError:
            raise ValueError("The provided dataset is empty (0 rows).")

    def _load_csv_buffer(self, buffer: Any) -> pd.DataFrame:
        encodings = ["utf-8", "latin1", "cp1252"]
        if hasattr(buffer, "seek"):
            buffer.seek(0)
            
        for enc in encodings:
            try:
                df = pd.read_csv(buffer, encoding=enc)
                if hasattr(buffer, "seek"):
                    buffer.seek(0)
                return df
            except pd.errors.EmptyDataError:
                raise ValueError("The provided dataset is empty (0 rows).")
            except Exception:
                if hasattr(buffer, "seek"):
                    buffer.seek(0)
                continue
                
        if hasattr(buffer, "seek"):
            buffer.seek(0)
        try:
            return pd.read_csv(buffer, errors="replace")
        except pd.errors.EmptyDataError:
            raise ValueError("The provided dataset is empty (0 rows).")

    def _parse_potential_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Heuristically convert string columns with date patterns to datetime without failing."""
        df = df.copy()
        date_keywords = ["date", "time", "timestamp", "year", "month", "day", "period"]
        for col in df.columns:
            col_lower = str(col).lower()
            is_str_col = pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object
            if is_str_col and any(k in col_lower for k in date_keywords):
                try:
                    converted = pd.to_datetime(df[col], errors="coerce")
                    if converted.notnull().mean() > 0.70:
                        df[col] = converted
                except Exception:
                    pass
        return df
