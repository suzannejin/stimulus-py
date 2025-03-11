"""This module provides classes for handling CSV data files in the STIMULUS format.

The module contains three main classes:
- DatasetHandler: Base class for loading and managing CSV data
- DatasetProcessor: Class for preprocessing data with transformations and splits
- DatasetLoader: Class for loading processed data for model training

The data format consists of:
1. A CSV file containing the raw data
2. A YAML configuration file that defines:
   - Column names and their roles (input/label/meta)
   - Data types and encoders for each column
   - Transformations to apply (noise, augmentation, etc.)
   - Split configuration for train/val/test sets

The data handling pipeline consists of:
1. Loading raw CSV data according to the YAML config
2. Applying configured transformations
3. Splitting into train/val/test sets based on config
4. Encoding data for model training using specified encoders

See titanic.yaml in tests/test_data/titanic/ for an example configuration file format.
"""

from typing import Any, Optional

import numpy as np
import polars as pl
import torch

import stimulus.data.encoding.encoders as encoders_module
import stimulus.data.splitting.splitters as splitters_module
import stimulus.data.transforming.transforms as transforms_module


class DatasetHandler:
    """Main class for handling dataset loading, encoding, transformation and splitting.

    This class coordinates the interaction between different managers to process
    CSV datasets according to the provided configuration.

    Attributes:
        encoder_manager (EncodeManager): Manager for handling data encoding operations.
        transform_manager (TransformManager): Manager for handling data transformations.
        split_manager (SplitManager): Manager for handling dataset splitting.
    """

    def __init__(
        self,
        csv_path: str,
    ) -> None:
        """Initialize the DatasetHandler with required config.

        Args:
            csv_path (str): Path to the CSV data file.
        """
        self.columns = self.read_csv_header(csv_path)
        self.data = self.load_csv(csv_path)

    def read_csv_header(self, csv_path: str) -> list:
        """Get the column names from the header of the CSV file.

        Args:
            csv_path (str): Path to the CSV file to read headers from.

        Returns:
            list: List of column names from the CSV header.
        """
        with open(csv_path) as f:
            return f.readline().strip().split(",")

    def select_columns(self, columns: list) -> dict:
        """Select specific columns from the DataFrame and return as a dictionary.

        Args:
            columns (list): List of column names to select.

        Returns:
            dict: A dictionary where keys are column names and values are lists containing the column data.

        Example:
            >>> handler = DatasetHandler(...)
            >>> data_dict = handler.select_columns(["col1", "col2"])
            >>> # Returns {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
        """
        df = self.data.select(columns)
        return {col: df[col].to_list() for col in columns}

    def load_csv(self, csv_path: str) -> pl.DataFrame:
        """Load the CSV file into a polars DataFrame.

        Args:
            csv_path (str): Path to the CSV file to load.

        Returns:
            pl.DataFrame: Polars DataFrame containing the loaded CSV data.
        """
        return pl.read_csv(csv_path)

    def save(self, path: str) -> None:
        """Saves the data to a csv file."""
        self.data.write_csv(path)


class DatasetProcessor(DatasetHandler):
    """Class for loading dataset, applying transformations and splitting."""

    def __init__(
        self,
        csv_path: str,
        transforms: dict[str, list[transforms_module.AbstractTransform]],
        split_columns: list[str],
        splitter: splitters_module.AbstractSplitter,
    ) -> None:
        """Initialize the DatasetProcessor."""
        super().__init__(csv_path)
        self.transforms = transforms
        self.split_columns = split_columns
        self.splitter = splitter

    def add_split(self, *, force: bool = False) -> None:
        """Add a column specifying the train, validation, test splits of the data.

        An error exception is raised if the split column is already present in the csv file. This behaviour can be overriden by setting force=True.

        Args:
            force (bool): If True, the split column present in the csv file will be overwritten.
        """
        if ("split" in self.columns) and (not force):
            raise ValueError(
                "The category split is already present in the csv file. If you want to still use this function, set force=True",
            )
        # get relevant split columns from the dataset_manager
        split_input_data = self.select_columns(self.split_columns)

        # get the split indices
        train, validation, test = self.splitter.get_split_indexes(split_input_data)

        # add the split column to the data
        split_column = np.full(len(self.data), -1).astype(int)
        split_column[train] = 0
        split_column[validation] = 1
        split_column[test] = 2
        self.data = self.data.with_columns(pl.Series("split", split_column))

        if "split" not in self.columns:
            self.columns.append("split")

    def apply_transformations(self) -> None:
        """Apply the transformation group to the data.

        Applies all transformations defined in self.transforms to their corresponding columns.
        Each column can have multiple transformations that are applied sequentially.
        """
        for column_name, transforms_list in self.transforms.items():
            for transform in transforms_list:
                transformed_data = transform.transform_all(self.data[column_name].to_list())

                if transform.add_row:
                    new_rows = self.data.with_columns(
                        pl.Series(column_name, transformed_data),
                    )
                    self.data = pl.vstack(self.data, new_rows)
                else:
                    self.data = self.data.with_columns(
                        pl.Series(column_name, transformed_data),
                    )

    def shuffle_labels(self, label_columns: list[str], seed: Optional[float] = None) -> None:
        """Shuffles the labels in the data."""
        # set the np seed
        np.random.seed(seed)

        for key in label_columns:
            self.data = self.data.with_columns(
                pl.Series(key, np.random.permutation(list(self.data[key]))),
            )


class DatasetLoader(DatasetHandler):
    """Class for loading dataset and passing it to the deep learning model."""

    def __init__(
        self,
        encoders: dict[str, encoders_module.AbstractEncoder],
        input_columns: list[str],
        label_columns: list[str],
        meta_columns: list[str],
        csv_path: str,
        split: Optional[int] = None,
    ) -> None:
        """Initialize the DatasetLoader."""
        super().__init__(csv_path)
        self.encoders = encoders
        self.data = self.load_csv_per_split(csv_path, split) if split is not None else self.load_csv(csv_path)
        self.input_columns = input_columns
        self.label_columns = label_columns
        self.meta_columns = meta_columns

    def encode_dataframe(self, dataframe: pl.DataFrame) -> dict[str, torch.Tensor]:
        """Encode the dataframe columns using the configured encoders.

        Takes a polars DataFrame and encodes each column using its corresponding encoder
        from self.encoders.

        Args:
            dataframe: Polars DataFrame containing the columns to encode

        Returns:
            Dict mapping column names to their encoded tensors. The exact shape of each
            tensor depends on the encoder used for that column.

        Example:
            >>> df = pl.DataFrame({"dna_seq": ["ACGT", "TGCA"], "labels": [1, 2]})
            >>> encoded = dataset_loader.encode_dataframe(df)
            >>> print(encoded["dna_seq"].shape)
            torch.Size([2, 4, 4])  # 2 sequences, length 4, one-hot encoded
        """
        return {col: self.encoders[col].encode_all(dataframe[col].to_list()) for col in dataframe.columns}

    def get_all_items(self) -> tuple[dict, dict, dict]:
        """Get the full dataset as three separate dictionaries for inputs, labels and metadata.

        Returns:
            tuple[dict, dict, dict]: Three dictionaries containing:
                - Input dictionary mapping input column names to encoded input data
                - Label dictionary mapping label column names to encoded label data
                - Meta dictionary mapping meta column names to meta data

        Example:
            >>> handler = DatasetHandler(...)
            >>> input_dict, label_dict, meta_dict = handler.get_dataset()
            >>> print(input_dict.keys())
            dict_keys(['age', 'fare'])
            >>> print(label_dict.keys())
            dict_keys(['survived'])
            >>> print(meta_dict.keys())
            dict_keys(['passenger_id'])
        """
        input_data = self.encode_dataframe(self.data[self.input_columns])
        label_data = self.encode_dataframe(self.data[self.label_columns])
        meta_data = {key: self.data[key].to_list() for key in self.meta_columns}
        return input_data, label_data, meta_data

    def get_all_items_and_length(self) -> tuple[tuple[dict, dict, dict], int]:
        """Get the full dataset as three separate dictionaries for inputs, labels and metadata, and the length of the data."""
        return self.get_all_items(), len(self.data)

    def load_csv_per_split(self, csv_path: str, split: int) -> pl.DataFrame:
        """Load the part of csv file that has the specified split value.

        Split is a number that for 0 is train, 1 is validation, 2 is test.
        """
        if "split" not in self.columns:
            raise ValueError("The category split is not present in the csv file")
        if split not in [0, 1, 2]:
            raise ValueError(
                f"The split value should be 0, 1 or 2. The specified split value is {split}",
            )
        return pl.scan_csv(csv_path).filter(pl.col("split") == split).collect()

    def __len__(self) -> int:
        """Return the length of the first list in input, assumes that all are the same length."""
        return len(self.data)

    def __getitem__(
        self,
        idx: Any,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, list]]:
        """Get the data at a given index, and encodes the input and label, leaving meta as it is.

        Args:
            idx: The index of the data to be returned, it can be a single index, a list of indexes or a slice
        """
        # Handle different index types
        if isinstance(idx, slice):
            # Get the actual indices for the slice
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else len(self.data)
            data_at_index = self.data.slice(start, stop - start)

            # Process DataFrame
            input_data = self.encode_dataframe(
                data_at_index[self.input_columns],
            )
            label_data = self.encode_dataframe(
                data_at_index[self.label_columns],
            )
            meta_data = {key: data_at_index[key].to_list() for key in self.meta_columns}

        elif isinstance(idx, int):
            # For single row, convert to dict with column names as keys
            row_dict = dict(zip(self.data.columns, self.data.row(idx)))

            # Create single-row DataFrames for encoding
            input_df = pl.DataFrame({col: [row_dict[col]] for col in self.input_columns})
            label_df = pl.DataFrame({col: [row_dict[col]] for col in self.label_columns})

            input_data = self.encode_dataframe(input_df)
            label_data = self.encode_dataframe(label_df)
            meta_data = {key: [row_dict[key]] for key in self.meta_columns}

        else:  # list or other sequence
            data_at_index = self.data.select(idx)

            # Process DataFrame
            input_data = self.encode_dataframe(
                data_at_index[self.input_columns],
            )
            label_data = self.encode_dataframe(
                data_at_index[self.label_columns],
            )
            meta_data = {key: data_at_index[key].to_list() for key in self.meta_columns}

        return input_data, label_data, meta_data


class TorchDataset(torch.utils.data.Dataset):
    """Class for creating a torch dataset."""

    def __init__(
        self,
        loader: DatasetLoader,
    ) -> None:
        """Initialize the TorchDataset.

        Args:
            loader: A DatasetLoader instance
        """
        self.loader = loader

    def __len__(self) -> int:
        return len(self.loader)

    def __getitem__(self, idx: int) -> tuple[dict, dict, dict]:
        return self.loader[idx]
