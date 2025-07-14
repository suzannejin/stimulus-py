"""This file contains encoders classes for encoding various types of data."""

import logging
from abc import ABC, abstractmethod
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from sklearn import preprocessing
from transformers import AutoModel, AutoTokenizer

from stimulus.learner.optuna_tune import get_device

logger = logging.getLogger(__name__)


class AbstractEncoder(ABC):
    """Abstract class for encoders.

    Encoders are classes that encode the raw data into torch.tensors.
    Different encoders provide different encoding methods.
    Different encoders may take different types of data as input.

    Methods:
        batch_encode: encodes a list of data points into a numpy.ndarray
    """

    @abstractmethod
    def batch_encode(self, data: np.ndarray) -> np.ndarray:
        """Encode a batch of data points.

        This is an abstract method, child classes should overwrite it.

        Args:
            data (np.ndarray): a batch of data points

        Returns:
            encoded_data (np.ndarray): encoded data points
        """
        raise NotImplementedError


class TextOneHotEncoder(AbstractEncoder):
    """One hot encoder for text data with highly optimized implementation.

    If a character c is not in the alphabet, c will be represented by a vector of zeros.
    This encoder is optimized for processing large batches of sequences efficiently on GPU.
    """

    # Constants to replace magic numbers
    TENSOR_3D_SHAPE = 3
    ASCII_MAX_VALUE = 128

    def __init__(
        self,
        alphabet: str = "acgt",
        dtype: Optional[np.dtype[np.floating]] = None,
        *,
        convert_lowercase: bool = False,
        force_cpu: bool = True,
        padding: bool = False,
    ) -> None:
        """Initialize the TextOneHotEncoder class.

        Args:
            alphabet (str): the alphabet to one hot encode the data with.
            dtype (np.dtype): the data type of the encoded data. Default = np.dtype(np.float32)
            convert_lowercase (bool): whether to convert sequences to lowercase.
            force_cpu (bool): whether to force the encoder to run on CPU.
            padding (bool): whether to pad sequences of different lengths.
        """
        if dtype is None:
            dtype = np.dtype(np.float32)
        if convert_lowercase:
            alphabet = alphabet.lower()
        self.convert_lowercase = convert_lowercase
        self.alphabet = alphabet
        self.padding = padding
        self.dtype = dtype
        self.device = get_device() if not force_cpu else torch.device("cpu")

        # Pre-compute and cache character mappings for the entire ASCII range
        self.UNKNOWN_IDX = -1
        self.alphabet_size = len(alphabet)

        # Build a fast ASCII-to-index mapping table directly on the GPU
        # This is more efficient than dictionary lookups
        self.lookup_table = torch.full((self.ASCII_MAX_VALUE,), self.UNKNOWN_IDX, dtype=torch.int64, device=self.device)

        # Fill the lookup table with character indices
        for idx, char in enumerate(alphabet):
            self.lookup_table[ord(char)] = idx
            # Handle case conversion if needed
            if convert_lowercase:
                if "A" <= char <= "Z":
                    self.lookup_table[ord(char.lower())] = idx
                elif "a" <= char <= "z":
                    self.lookup_table[ord(char.upper())] = idx

        # Pre-allocate a mask for invalid characters to avoid nonzero operations
        # Initialize with ones to mark valid positions by default
        torch_dtype = (
            torch.float32
            if dtype == np.dtype(np.float32)
            else torch.float64
            if dtype == np.dtype(np.float64)
            else torch.float32
        )
        self.alphabet_mask = torch.ones(self.alphabet_size + 1, dtype=torch_dtype, device=self.device)
        # Set the last position (for invalid characters) to zero
        self.alphabet_mask[-1] = 0.0

    def batch_encode(self, data: np.ndarray) -> np.ndarray:
        """Encode all sequences in a batch using fully vectorized operations.

        Args:
            data (np.ndarray): A 1D numpy array of strings (sequences).

        Returns:
            np.ndarray: Array of shape (batch_size, max_seq_length, alphabet_size)
        """
        # Handle single string case by ensuring data is a 1D array
        if data.ndim == 0:  # handles case where a single string is passed as a 0-d array
            data = np.array([str(data)])
        elif not (data.ndim == 1 and data.dtype.kind in ["U", "S"]):
            error_msg = (
                f"Expected 1D numpy array of strings for data, got array with shape {data.shape} and dtype {data.dtype}"
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        # Early check for sequence length consistency when padding=False
        if not self.padding:
            lengths = {len(seq) for seq in data}
            if len(lengths) > 1:
                error_msg = "All sequences must have the same length when padding is False."
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Find max length for processing all sequences at once
        # Ensure that data is not empty before calling max()
        if data.size == 0:
            return np.array([]).reshape(0, 0, self.alphabet_size)  # Or handle as an error

        max_length = max(len(seq) for seq in data) if data.size > 0 else 0
        batch_size = len(data)

        # OPTIMIZATION: Process all sequences as a single byte array
        # This eliminates Python loops and character-by-character processing
        ascii_array = np.zeros((batch_size, max_length), dtype=np.uint8)

        # Convert sequences to bytes more efficiently
        for i, seq_input_bytes in enumerate(data):
            seq_input = seq_input_bytes.decode("utf-8") if isinstance(seq_input_bytes, bytes) else str(seq_input_bytes)
            seq = seq_input.lower() if self.convert_lowercase else seq_input

            # OPTIMIZATION: Use numpy byte array conversion to avoid Python loop
            seq_bytes = np.frombuffer(seq.encode("ascii", errors="ignore"), dtype=np.uint8)
            ascii_array[i, : len(seq_bytes)] = seq_bytes

        # Transfer to GPU in one operation
        # OPTIMIZATION: Use torch.tensor directly on device rather than to() to avoid copy
        ascii_tensor = torch.tensor(ascii_array, dtype=torch.int64, device=self.device)
        # OPTIMIZATION: Create valid ASCII mask directly
        # This combines multiple operations into one
        valid_mask = (ascii_tensor > 0) & (ascii_tensor < self.ASCII_MAX_VALUE)

        # Create indices tensor - use -1 for padding/invalid chars
        indices = torch.full_like(ascii_tensor, self.UNKNOWN_IDX)

        # Only lookup valid ASCII values (avoiding unnecessary computation)
        valid_indices = valid_mask.nonzero(as_tuple=True)
        indices[valid_indices] = self.lookup_table[ascii_tensor[valid_indices]]

        # For one-hot encoding, we need non-negative indices
        # OPTIMIZATION: Use a single mask for padding and unknown chars
        valid_indices_mask = indices >= 0
        safe_indices = indices.clone()
        safe_indices[~valid_indices_mask] = 0  # Temporary index for one_hot

        # Apply one-hot encoding - FIX: removed the 'out' parameter
        torch_dtype = (
            torch.float32
            if self.dtype == np.dtype(np.float32)
            else torch.float64
            if self.dtype == np.dtype(np.float64)
            else torch.float32
        )
        one_hot = F.one_hot(safe_indices, num_classes=self.alphabet_size + 1).to(torch_dtype)

        # Apply alphabet mask to zero out invalid indices
        # This creates zeros for unknown characters
        result = one_hot.clone()
        result[~valid_indices_mask] = 0.0

        # Remove the last dimension (sentinel value) to get the final shape
        return result[:, :, : self.alphabet_size].cpu().numpy().astype(self.dtype)


class TextAsciiEncoder(AbstractEncoder):
    """Encoder for text data that encodes the data based on ASCII values.

    Attributes:
        vocab_size (int): The size of the vocabulary. Default = 256 (ASCII characters)
        dtype (np.dtype): The data type of the encoded data. Default = np.dtype(np.int8)
        max_len (Optional[int]): the length to pad the sequences to. No padding is done if set to None. Default = None
        trim_strategy (Literal["raise", "trim", "slice", "drop"]): Behavior when a string is longer than max_len. Default = "raise"

    Methods:
        batch_encode: encodes a list of data points into a numpy.ndarray
    """

    def __init__(
        self,
        vocab_size: int = 256,
        dtype: Optional[np.dtype[np.signedinteger]] = None,
        *,
        max_len: Optional[int] = None,
        trim_strategy: Literal["raise", "trim", "slice", "drop"] = "raise",
    ) -> None:
        """Initialize the TextAsciiEncoder class.

        Args:
            vocab_size (int): the size of the vocabulary. Default = 256 (ASCII characters)
            dtype (np.dtype): the data type of the encoded data. Default = np.dtype(np.int8)
            max_len (Optional[int]): the length to pad the sequences to. No padding is done if set to None. Default = None
            trim_strategy (Literal["raise", "trim", "slice", "drop"]): Behavior when a string is longer than max_len. Default = "raise"

        Raises:
            ValueError: If an invalid trim strategy is provided.
        """
        if dtype is None:
            dtype = np.dtype(np.int8)
        self.vocab_size = vocab_size
        self.dtype = dtype
        self.max_len = max_len
        self.trim_strategy = trim_strategy
        if self.trim_strategy not in ["raise", "trim", "slice", "drop"]:
            raise ValueError(
                f"Invalid trim strategy: {self.trim_strategy}",
            )

    def batch_encode(self, data: np.ndarray) -> np.ndarray:
        """Encodes the data.

        This method takes as input a 1D numpy array of strings and returns a numpy array.

        Args:
            data (np.ndarray): a 1D numpy array of strings

        Returns:
            encoded_data (np.ndarray): the encoded data

        Raises:
            TypeError: If the input data is not a 1D numpy array of strings.
            ValueError: If any string in data contains characters with ASCII values greater than vocab_size - 1
        """
        if not (isinstance(data, np.ndarray) and data.ndim == 1 and data.dtype.kind in ["U", "S"]):
            raise TypeError(
                f"Expected input data to be a 1D numpy array of strings, got {type(data).__name__} with dtype {data.dtype if hasattr(data, 'dtype') else 'N/A'}",
            )

        encoded_data_list = []
        for s_bytes in data:
            s = s_bytes.decode("utf-8") if isinstance(s_bytes, bytes) else str(s_bytes)  # Ensure it's a string
            if any(ord(c) >= self.vocab_size for c in s):
                raise ValueError(
                    f"Data string '{s}' contains characters with ASCII values greater than {self.vocab_size - 1}",
                )

            values = np.frombuffer(s.encode("ascii", errors="ignore"), dtype=np.uint8)

            current_max_len = self.max_len
            if current_max_len is None:  # If no global max_len, use the length of the current string
                current_max_len = len(values)

            if len(values) > current_max_len:
                if self.trim_strategy == "raise":
                    # raise an error and terminate
                    raise ValueError(
                        f"Data length {len(values)} is greater than the specified max_len {current_max_len}",
                    )
                if self.trim_strategy == "trim":
                    # trim the array to max_len, discard the overflow
                    logger.warning(f"Trimming an item of length {len(values)}")
                    encoded_data_list.append(values[:current_max_len])
                elif self.trim_strategy == "slice":
                    # split items that are too long
                    logger.warning(f"Slicing an item of length {len(values)}")
                    num_chunks = len(values) // current_max_len + (1 if len(values) % current_max_len != 0 else 0)
                    for i in range(num_chunks):
                        chunk = values[i * current_max_len : (i + 1) * current_max_len]
                        padded_chunk = np.pad(chunk, (0, current_max_len - len(chunk)), mode="constant")
                        encoded_data_list.append(padded_chunk)
                elif self.trim_strategy == "drop":
                    # skip the offending item
                    logger.warning(f"Skipping an item of length {len(values)}")
                    continue
                else:
                    # somehow the trim strategy is wrong
                    raise ValueError(
                            "Trim strategy is invalid in batch_encode. Please check your code for manual updates of this attribute.",
                    )
            else:
                # Pad the single array/chunk
                padded_values = np.pad(values, (0, current_max_len - len(values)), mode="constant")
                encoded_data_list.append(padded_values)

        if not encoded_data_list:  # Handle empty input data
            return np.array([], dtype=self.dtype)

        return np.array(encoded_data_list, dtype=self.dtype)


class NumericEncoder(AbstractEncoder):
    """Encoder for float/int data.

    Attributes:
        dtype (np.dtype): The data type of the encoded data. Default = np.dtype(np.float32)
    """

    def __init__(self, dtype: Optional[np.dtype[np.number]] = None) -> None:
        """Initialize the NumericEncoder class.

        Args:
            dtype (np.dtype): the data type of the encoded data. Default = np.dtype(np.float32)
        """
        if dtype is None:
            dtype = np.dtype(np.float32)
        self.dtype = dtype

    def batch_encode(self, data: np.ndarray) -> np.ndarray:
        """Encodes the data.

        This method takes as input a 1D numpy array of numbers and returns a numpy array.

        Args:
            data (np.ndarray): a 1D numpy array of numbers

        Returns:
            encoded_data (np.ndarray): the encoded data
        """
        if not isinstance(data, np.ndarray):  # Check if it's a numpy array first
            data = np.array(data)  # Convert if it's a list or other compatible type

        return data.astype(self.dtype)


class StrClassificationEncoder(AbstractEncoder):
    """A string classification encoder that converts lists of strings into numeric labels using scikit-learn.

    When scale is set to True, the labels are scaled to be between 0 and 1.

    Attributes:
        scale (bool): Whether to scale the labels to be between 0 and 1. Default = False
        dtype (np.dtype): The data type of the encoded data. Default = np.dtype(np.int16)

    Methods:
        batch_encode: encodes a list of data points into a numpy.ndarray
    """

    def __init__(self, *, scale: bool = False, dtype: Optional[np.dtype[np.signedinteger]] = None) -> None:
        """Initialize the StrClassificationEncoder class.

        Args:
            scale (bool): whether to scale the labels to be between 0 and 1. Default = False
            dtype (np.dtype): the data type of the encoded data. Default = np.dtype(np.int16)
        """
        if dtype is None:
            dtype = np.dtype(np.int16)
        self.scale = scale
        self.dtype = dtype

    def batch_encode(self, data: np.ndarray) -> np.ndarray:
        """Encodes the data.

        This method takes as input a 1D numpy array of strings,
        should be mappable to a single output, using LabelEncoder from scikit learn and returning a numpy array.
        For more info visit : https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

        Args:
            data (np.ndarray): a 1D numpy array of strings

        Returns:
            encoded_data (np.ndarray): the encoded data
        """
        if not (
            isinstance(data, np.ndarray) and data.ndim == 1 and data.dtype.kind in ["U", "S"]
        ):  # Check for 1D array of strings
            raise TypeError(
                f"Expected input data to be a 1D numpy array of strings, got {type(data).__name__} with dtype {data.dtype if hasattr(data, 'dtype') else 'N/A'}",
            )

        self._check_dtype(data)

        encoder = preprocessing.LabelEncoder()
        # scikit-learn's LabelEncoder expects a list or 1D array-like of strings
        encoded_data_np = encoder.fit_transform(data)
        if self.scale:
            encoded_data_np = encoded_data_np / max(len(encoded_data_np) - 1, 1)

        return encoded_data_np.astype(self.dtype)

    def _check_dtype(self, data: np.ndarray) -> None:
        """Check if the input data is string data.

        Args:
            data (np.ndarray): a 1D numpy array of strings

        Raises:
            ValueError: If the input data is not a 1D numpy array of strings
        """
        if not (data.ndim == 1 and data.dtype.kind in ["U", "S"]):
            err_msg = "Expected input data to be a 1D numpy array of strings"
            logger.error(err_msg)
            raise ValueError(err_msg)


class NumericRankEncoder(AbstractEncoder):
    """Encoder for float/int data that encodes the data based on their rank.

    Attributes:
        scale (bool): whether to scale the ranks to be between 0 and 1. Default = False
        dtype (np.dtype): The data type of the encoded data. Default = np.dtype(np.int16)

    Methods:
        batch_encode: encodes a list of data points into a numpy.ndarray
    """

    def __init__(self, *, scale: bool = False, dtype: Optional[np.dtype[np.signedinteger]] = None) -> None:
        """Initialize the NumericRankEncoder class.

        Args:
            scale (bool): whether to scale the ranks to be between 0 and 1. Default = False
            dtype (np.dtype): the data type of the encoded data. Default = np.dtype(np.int16)
        """
        if dtype is None:
            dtype = np.dtype(np.int16)
        self.scale = scale
        self.dtype = dtype

    def batch_encode(self, data: np.ndarray) -> np.ndarray:
        """Encodes the data.

        This method takes as input a 1D numpy array of numbers, and returns the ranks of the data points.
        The ranks are normalized to be between 0 and 1, when scale is set to True.

        Args:
            data (np.ndarray): a 1D numpy array of numeric values

        Returns:
            encoded_data (np.ndarray): the encoded data
        """
        if not isinstance(data, np.ndarray):  # Check if it's a numpy array
            data = np.array(data)  # Convert if it's a list or other compatible type

        self._check_input_dtype(data)

        # Get ranks (0 is lowest, n-1 is highest)
        # and normalize to be between 0 and 1
        ranks: np.ndarray = np.argsort(np.argsort(data))
        if self.scale:
            ranks = ranks / max(len(ranks) - 1, 1)

        return ranks.astype(self.dtype)

    def _check_input_dtype(self, data: np.ndarray) -> None:
        """Check if the input data is int or float data.

        Args:
            data (np.ndarray): a 1D numpy array of numeric values

        Raises:
            ValueError: If the input data is not numeric
        """
        if not np.issubdtype(data.dtype, np.number):
            err_msg = f"Expected input data to be numeric (int or float), got dtype {data.dtype}"
            logger.error(err_msg)
            raise ValueError(err_msg)


class HuggingFaceEmbeddingEncoder(AbstractEncoder):
    """HuggingFace embedding encoder using CLS token from last layer."""

    def __init__(
        self,
        model_repo_name: str,
        batch_size: int = 32,
        dtype: Optional[np.dtype[np.floating]] = None,
        layer_index: int = -1,
    ) -> None:
        """Initialize encoder with HuggingFace model.

        Args:
            model_repo_name: HuggingFace model repository name
            batch_size: Batch size for processing
            dtype: Output data type
            layer_index: Which hidden layer to use (-1 for last, 0 for first, etc.)
        """
        if dtype is None:
            dtype = np.dtype(np.float32)

        self.batch_size = batch_size
        self.dtype = dtype
        self.layer_index = layer_index
        self.device = get_device()

        # Load model with output_hidden_states=True to access all layers
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo_name)
        self.model = AutoModel.from_pretrained(model_repo_name)
        self.model.to(self.device)
        self.model.eval()

    def batch_encode(self, data: np.ndarray) -> np.ndarray:
        """Encode sequences to embeddings using CLS token."""
        if data.size == 0:
            return np.array([]).reshape(0, 0)

        # Convert to string list
        sequences = [str(s.decode("utf-8") if isinstance(s, bytes) else s) for s in data]

        all_embeddings = []
        for i in range(0, len(sequences), self.batch_size):
            batch_seq = sequences[i : i + self.batch_size]

            # Tokenize and encode
            inputs = self.tokenizer(batch_seq, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Select layer and use CLS token (first token)
                hidden_states = outputs.hidden_states[self.layer_index]
                cls_embeddings = hidden_states[:, 0, :]

            all_embeddings.append(cls_embeddings.cpu().numpy().astype(self.dtype))

        return np.concatenate(all_embeddings, axis=0)
