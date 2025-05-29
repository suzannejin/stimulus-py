"""Test for encoders."""

import numpy as np
import pytest
import torch

from src.stimulus.data.encoding.encoders import (
    NumericEncoder,
    NumericRankEncoder,
    StrClassificationEncoder,
    TextAsciiEncoder,
    TextOneHotEncoder,
)


class TestTextOneHotEncoder:
    """Test suite for TextOneHotEncoder."""

    @staticmethod
    @pytest.fixture
    def encoder_default() -> TextOneHotEncoder:
        """Provide a default encoder.

        Returns:
            TextOneHotEncoder: A default encoder instance
        """
        return TextOneHotEncoder(alphabet="acgt", padding=True)

    @staticmethod
    @pytest.fixture
    def encoder_lowercase() -> TextOneHotEncoder:
        """Provide an encoder with convert_lowercase set to True.

        Returns:
            TextOneHotEncoder: An encoder instance with lowercase conversion
        """
        return TextOneHotEncoder(alphabet="ACgt", convert_lowercase=True, padding=True)

    # ---- Test for initialization ---- #

    def test_init_with_string_alphabet(self) -> None:
        """Test initialization with valid string alphabet."""
        encoder = TextOneHotEncoder(alphabet="acgt")
        assert encoder.alphabet == "acgt"
        assert encoder.convert_lowercase is False
        assert encoder.padding is False

    # ---- Tests for batch_encode ---- #

    def test_batch_encode_returns_numpy_array(self, encoder_default: TextOneHotEncoder) -> None:
        """Test that batch_encode returns a numpy array of the correct shape."""
        seq = np.array(["acgt"])
        encoded = encoder_default.batch_encode(seq)
        assert isinstance(encoded, np.ndarray)
        # shape should be (batch_size=1, len(seq)=4, alphabet_size=4)
        assert encoded.shape == (1, 4, 4)

    def test_batch_encode_unknown_character_returns_zero_vector(self, encoder_default: TextOneHotEncoder) -> None:
        """Test that encoding an unknown character returns a zero vector."""
        seq = np.array(["acgtn"])
        encoded = encoder_default.batch_encode(seq)
        # the last character 'n' is not in 'acgt', so the last row should be all zeros
        assert np.all(encoded[0, -1] == 0)

    def test_batch_encode_default(self, encoder_default: TextOneHotEncoder) -> None:
        """Test case-sensitive encoding behavior.

        Case-sensitive: 'ACgt' => 'ACgt' means 'A' and 'C' are uppercase in the alphabet,
        'g' and 't' are lowercase in the alphabet.
        """
        seq = np.array(["ACgt"])
        encoded = encoder_default.batch_encode(seq)
        # shape = (1, len(seq)=4, 4)
        assert encoded.shape == (1, 4, 4)
        # 'A' should be one-hot at the 0th index, 'C' at the 1st index, 'g' at the 2nd, 't' at the 3rd.
        # The order of categories in OneHotEncoder is typically ['A', 'C', 'g', 't'] given we passed ['A','C','g','t'].
        assert np.all(encoded[0, 0] == np.array([0, 0, 0, 0]))  # 'A'
        assert np.all(encoded[0, 1] == np.array([0, 0, 0, 0]))  # 'C'
        assert np.all(encoded[0, 2] == np.array([0, 0, 1, 0]))  # 'g'
        assert np.all(encoded[0, 3] == np.array([0, 0, 0, 1]))  # 't'

    def test_batch_encode_lowercase(self, encoder_lowercase: TextOneHotEncoder) -> None:
        """Case-insensitive: 'ACgt' => 'acgt' internally."""
        seq = np.array(["ACgt"])
        encoded = encoder_lowercase.batch_encode(seq)
        # shape = (1, 4, 4)
        assert encoded.shape == (1, 4, 4)
        # The order of categories in OneHotEncoder is typically ['a', 'c', 'g', 't'] for the default encoder.
        assert np.all(encoded[0, 0] == np.array([1, 0, 0, 0]))  # 'a'
        assert np.all(encoded[0, 1] == np.array([0, 1, 0, 0]))  # 'c'
        assert np.all(encoded[0, 2] == np.array([0, 0, 1, 0]))  # 'g'
        assert np.all(encoded[0, 3] == np.array([0, 0, 0, 1]))  # 't'

    def test_batch_encode_with_single_string(self, encoder_default: TextOneHotEncoder) -> None:
        """Test encoding a single string with batch_encode."""
        seq = np.array(["acgt"])
        encoded = encoder_default.batch_encode(seq)
        # shape = (batch_size=1, seq_len=4, alphabet_size=4)
        assert encoded.shape == (1, 4, 4)

    def test_batch_encode_with_list_of_sequences(self, encoder_default: TextOneHotEncoder) -> None:
        """Test encoding multiple sequences with batch_encode."""
        seqs = np.array(["acgt", "acgtn"])  # second has an unknown 'n'
        encoded = encoder_default.batch_encode(seqs)
        # shape = (2, max_len=5, alphabet_size=4)
        assert encoded.shape == (2, 5, 4)

    def test_batch_encode_with_padding_false(self) -> None:
        """Test that batch_encode raises error when padding is False and sequences have different lengths."""
        encoder = TextOneHotEncoder(alphabet="acgt", padding=False)
        seqs = np.array(["acgt", "acgtn"])  # different lengths
        with pytest.raises(ValueError, match="All sequences must have the same length when padding is False."):
            encoder.batch_encode(seqs)


class TestTextAsciiEncoder:
    """Test suite for TextAsciiEncoder."""

    def test_batch_encode_single_string(self) -> None:
        """Test encoding a single string."""
        encoder = TextAsciiEncoder()
        input_str = np.array(["hello"])
        output = encoder.batch_encode(input_str)
        assert isinstance(output, np.ndarray)
        assert output.shape == (1, 5)
        assert np.array_equal(output[0], np.array([104, 101, 108, 108, 111]))

    def test_batch_encode_multiple_strings(self) -> None:
        """Test encoding a list of strings."""
        encoder = TextAsciiEncoder()
        input_strs = np.array(["hello", "world"])
        output = encoder.batch_encode(input_strs)
        assert isinstance(output, np.ndarray)
        assert output.shape == (2, 5)
        assert np.array_equal(output[0], np.array([104, 101, 108, 108, 111]))
        assert np.array_equal(output[1], np.array([119, 111, 114, 108, 100]))

    def test_batch_encode_padding(self) -> None:
        """Test encoding a list of strings with padding."""
        encoder = TextAsciiEncoder(max_len=10)
        input_strs = np.array(["hello", "worlds"])
        output = encoder.batch_encode(input_strs)
        assert isinstance(output, np.ndarray)
        assert output.shape == (2, 10)
        assert np.array_equal(output[0], np.array([104, 101, 108, 108, 111, 0, 0, 0, 0, 0]))
        assert np.array_equal(output[1], np.array([119, 111, 114, 108, 100, 115, 0, 0, 0, 0]))

    def test_batch_encode_dtype(self) -> None:
        """Test encoding with a non-default dtype."""
        encoder = TextAsciiEncoder(dtype=torch.int32)
        input_str = np.array(["hello"])
        output = encoder.batch_encode(input_str)
        assert output.dtype == np.int32

    def test_batch_encode_not_string_raises(self) -> None:
        """Test that encoding a non-string array raises a TypeError."""
        encoder = TextAsciiEncoder()
        with pytest.raises(TypeError):
            encoder.batch_encode(np.array([42]))  # type: ignore[arg-type]

    def test_batch_encode_unicode_raises(self) -> None:
        """Test that encoding a unicode string raises a ValueError."""
        encoder = TextAsciiEncoder()
        with pytest.raises(ValueError, match="Data string .* contains characters with ASCII values greater.*"):
            encoder.batch_encode(np.array(["你好"]))

    def test_batch_encode_too_long_raises(self) -> None:
        """Test that encoding a string that is too long raises a ValueError."""
        encoder = TextAsciiEncoder(max_len=3)
        with pytest.raises(ValueError, match="Data length .* is greater than the specified max_len.*"):
            encoder.batch_encode(np.array(["hello"]))


class TestNumericEncoder:
    """Test suite for NumericEncoder."""

    @staticmethod
    @pytest.fixture
    def float_encoder() -> NumericEncoder:
        """Provide a NumericEncoder instance.

        Returns:
            NumericEncoder: Default encoder instance
        """
        return NumericEncoder()

    @staticmethod
    @pytest.fixture
    def int_encoder() -> NumericEncoder:
        """Provide a NumericEncoder instance with integer dtype.

        Returns:
            NumericEncoder: Integer-based encoder instance
        """
        return NumericEncoder(dtype=torch.int32)

    def test_batch_encode_single_float(self, float_encoder: NumericEncoder) -> None:
        """Test encoding a single float value."""
        input_val = np.array([3.14])
        output = float_encoder.batch_encode(input_val)
        assert isinstance(output, np.ndarray), "Output should be a numpy array."
        assert output.dtype == np.float32, "Array dtype should be float32."
        assert output.size == 1, "Array should have exactly one element."
        assert output[0] == pytest.approx(3.14), "Encoded value does not match."

    def test_batch_encode_single_int(self, int_encoder: NumericEncoder) -> None:
        """Test encoding a single int value."""
        input_val = np.array([3])
        output = int_encoder.batch_encode(input_val)
        assert isinstance(output, np.ndarray), "Output should be a numpy array."
        assert output.dtype == np.int32, "Array dtype should be int32."
        assert output.size == 1, "Array should have exactly one element."
        assert output[0] == 3

    @pytest.mark.parametrize("fixture_name", ["float_encoder", "int_encoder"])
    def test_batch_encode_non_numeric_raises(
        self,
        request: pytest.FixtureRequest,
        fixture_name: str,
    ) -> None:
        """Test that encoding a non-numeric array raises a ValueError."""
        numeric_encoder = request.getfixturevalue(fixture_name)
        with pytest.raises(ValueError, match="Expected input data to be numeric"):
            numeric_encoder.batch_encode(np.array(["not_numeric"]))

    def test_batch_encode_multi_float(self, float_encoder: NumericEncoder) -> None:
        """Test batch_encode with a list of floats."""
        input_vals = np.array([3.14, 4.56])
        output = float_encoder.batch_encode(input_vals)
        assert isinstance(output, np.ndarray), "Output should be a numpy array."
        assert output.dtype == np.float32, "Array dtype should be float32."
        assert output.size == 2, "Array should have two elements."
        assert output[0] == pytest.approx(3.14), "First element does not match."
        assert output[1] == pytest.approx(4.56), "Second element does not match."

    def test_batch_encode_multi_int(self, int_encoder: NumericEncoder) -> None:
        """Test batch_encode with a list of integers."""
        input_vals = np.array([3, 4])
        output = int_encoder.batch_encode(input_vals)
        assert isinstance(output, np.ndarray), "Output should be a numpy array."
        assert output.dtype == np.int32, "Array dtype should be int32."
        assert output.size == 2, "Array should have two elements."
        assert output[0] == 3, "First element does not match."
        assert output[1] == 4, "Second element does not match."


class TestStrClassificationEncoder:
    """Test suite for StrClassificationIntEncoder and StrClassificationScaledEncoder."""

    @staticmethod
    @pytest.fixture
    def str_encoder() -> StrClassificationEncoder:
        """Provide a StrClassificationEncoder instance.

        Returns:
            StrClassificationEncoder: Default encoder instance
        """
        return StrClassificationEncoder(dtype=torch.int64)

    @staticmethod
    @pytest.fixture
    def scaled_encoder() -> StrClassificationEncoder:
        """Provide a StrClassificationEncoder with scaling enabled.

        Returns:
            StrClassificationEncoder: Scaled encoder instance
        """
        return StrClassificationEncoder(scale=True, dtype=torch.float32)

    @pytest.mark.parametrize(
        ("fixture", "expected_values"),
        [
            ("str_encoder", [0, 1, 2]),
            ("scaled_encoder", [0.0, 0.5, 1.0]),
        ],
    )
    def test_batch_encode_list_of_strings(
        self,
        request: pytest.FixtureRequest,
        fixture: str,
        expected_values: list,
    ) -> None:
        """Test encoding multiple unique strings.

        Verifies that the encoder produces correct array shape and values.
        """
        encoder = request.getfixturevalue(fixture)
        input_data = np.array(["apple", "banana", "cherry"])
        output = encoder.batch_encode(input_data)
        assert isinstance(output, np.ndarray)
        assert output.shape == (3,)
        assert np.allclose(output, np.array(expected_values))

    @pytest.mark.parametrize("fixture", ["str_encoder", "scaled_encoder"])
    def test_batch_encode_raises_type_error_on_non_string(
        self,
        request: pytest.FixtureRequest,
        fixture: str,
    ) -> None:
        """Tests that batch_encode raises TypeError if the input is not a numpy array of strings."""
        encoder = request.getfixturevalue(fixture)
        input_data = np.array([42])  # Not strings
        with pytest.raises(TypeError, match="Expected input data to be a 1D numpy array of strings"):
            encoder.batch_encode(input_data)


class TestNumericRankEncoder:
    """Test suite for NumericRankEncoder."""

    @staticmethod
    @pytest.fixture
    def rank_encoder() -> NumericRankEncoder:
        """Provide a NumericRankEncoder instance.

        Returns:
            NumericRankEncoder: Default encoder instance
        """
        return NumericRankEncoder()

    @staticmethod
    @pytest.fixture
    def scaled_encoder() -> NumericRankEncoder:
        """Provide a NumericRankEncoder with scaling enabled.

        Returns:
            NumericRankEncoder: Scaled encoder instance
        """
        return NumericRankEncoder(scale=True, dtype=torch.float32)

    def test_batch_encode_with_valid_rank(self, rank_encoder: NumericRankEncoder) -> None:
        """Test encoding a list of float values.

        Args:
            rank_encoder: Default rank encoder instance
        """
        input_vals = np.array([3.14, 2.71, 1.41])
        output = rank_encoder.batch_encode(input_vals)
        assert isinstance(output, np.ndarray), "Output should be a numpy array."
        assert output.size == 3, "Array should have exactly three elements."
        assert output[0] == 2, "First encoded value does not match."
        assert output[1] == 1, "Second encoded value does not match."
        assert output[2] == 0, "Third encoded value does not match."

    def test_batch_encode_with_valid_scaled_rank(self, scaled_encoder: NumericRankEncoder) -> None:
        """Test encoding a list of float values."""
        input_vals = np.array([3.14, 2.71, 1.41])
        output = scaled_encoder.batch_encode(input_vals)
        assert isinstance(output, np.ndarray), "Output should be a numpy array."
        assert output.size == 3, "Array should have exactly three elements."
        assert output[0] == pytest.approx(1), "First encoded value does not match."
        assert output[1] == pytest.approx(0.5), "Second encoded value does not match."
        assert output[2] == pytest.approx(0), "Third encoded value does not match."

    @pytest.mark.parametrize("fixture", ["rank_encoder", "scaled_encoder"])
    def test_batch_encode_with_non_numeric_raises(
        self,
        request: pytest.FixtureRequest,
        fixture: str,
    ) -> None:
        """Test that encoding a non-numeric array raises a ValueError.

        Args:
            request: Pytest fixture request
            fixture: Name of the fixture to use
        """
        encoder = request.getfixturevalue(fixture)
        with pytest.raises(ValueError, match="Expected input data to be numeric"):
            encoder.batch_encode(np.array(["not_numeric"]))
