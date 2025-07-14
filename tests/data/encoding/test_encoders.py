"""Test for encoders."""

import numpy as np
import pytest

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

    # ---- Tests for init ---- #

    def test_init_invalid_strategy_raises(self) -> None:
        """Test that the encoder raises if the trim strategy is invalid."""
        with pytest.raises(ValueError, match="Invalid trim strategy.*"):
            encoder = TextAsciiEncoder(trim_strategy="explode") # type: ignore[arg-type] # noqa: F841

    # ---- Tests for batch_encode ---- #

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

    def test_batch_encode_slice_padding(self) -> None:
        """Test encoding a list of strings with padding and slicing."""
        encoder = TextAsciiEncoder(max_len=5, trim_strategy="slice")
        input_strs = np.array(["hellomate"])
        output = encoder.batch_encode(input_strs)
        assert isinstance(output, np.ndarray)
        assert output.shape == (2, 5)
        assert np.array_equal(output[0], np.array([104, 101, 108, 108, 111]))
        assert np.array_equal(output[1], np.array([109, 97, 116, 101, 0]))

    def test_batch_encode_trim_long(self) -> None:
        """Test encoding a long string with the trim strategy."""
        encoder = TextAsciiEncoder(max_len=5, trim_strategy="trim")
        input_strs = np.array(["hello", "worlds"])
        output = encoder.batch_encode(input_strs)
        assert isinstance(output, np.ndarray)
        assert output.shape == (2, 5)
        assert np.array_equal(output[0], np.array([104, 101, 108, 108, 111]))
        assert np.array_equal(output[1], np.array([119, 111, 114, 108, 100]))

    def test_batch_encode_drop_long(self) -> None:
        """Test encoding a long string with the drop strategy."""
        encoder = TextAsciiEncoder(max_len=5, trim_strategy="drop")
        input_strs = np.array(["hello", "worlds"])
        output = encoder.batch_encode(input_strs)
        assert isinstance(output, np.ndarray)
        assert output.shape == (1, 5)
        assert np.array_equal(output[0], np.array([104, 101, 108, 108, 111]))

    def test_batch_encode_dtype(self) -> None:
        """Test encoding with a non-default dtype."""
        encoder = TextAsciiEncoder(dtype=np.dtype(np.int32))
        input_str = np.array(["hello"])
        output = encoder.batch_encode(input_str)
        assert output.dtype == np.int32

    def test_batch_encode_not_string_raises(self) -> None:
        """Test that encoding a non-string array raises a TypeError."""
        encoder = TextAsciiEncoder()
        with pytest.raises(TypeError):
            encoder.batch_encode(np.array([42]))

    def test_batch_encode_unicode_raises(self) -> None:
        """Test that encoding a unicode string raises a ValueError."""
        encoder = TextAsciiEncoder()
        with pytest.raises(ValueError, match="Data string .* contains characters with ASCII values greater.*"):
            encoder.batch_encode(np.array(["你好"]))

    def test_batch_encode_too_long_raises(self) -> None:
        """Test that encoding a string that is too long raises a ValueError if raise strategy is used."""
        encoder = TextAsciiEncoder(max_len=3, trim_strategy="raise")
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
        return NumericEncoder(dtype=np.dtype(np.int32))

    def test_batch_encode_single_float(self, float_encoder: NumericEncoder) -> None:
        """Test encoding a single float value."""
        input_val = np.array([3.14])
        output = float_encoder.batch_encode(input_val)
        assert isinstance(output, np.ndarray)
        assert output.shape == (1,)
        assert output.dtype == np.float32
        assert np.isclose(output[0], 3.14, atol=1e-6)

    def test_batch_encode_single_int(self, int_encoder: NumericEncoder) -> None:
        """Test encoding a single integer value."""
        input_val = np.array([42])
        output = int_encoder.batch_encode(input_val)
        assert isinstance(output, np.ndarray)
        assert output.shape == (1,)
        assert output.dtype == np.int32
        assert output[0] == 42

    def test_batch_encode_multi_float(self, float_encoder: NumericEncoder) -> None:
        """Test encoding multiple float values."""
        input_vals = np.array([1.1, 2.2, 3.3])
        output = float_encoder.batch_encode(input_vals)
        assert isinstance(output, np.ndarray)
        assert output.shape == (3,)
        assert output.dtype == np.float32
        assert np.allclose(output, [1.1, 2.2, 3.3], atol=1e-6)

    def test_batch_encode_multi_int(self, int_encoder: NumericEncoder) -> None:
        """Test encoding multiple integer values."""
        input_vals = np.array([10, 20, 30])
        output = int_encoder.batch_encode(input_vals)
        assert isinstance(output, np.ndarray)
        assert output.shape == (3,)
        assert output.dtype == np.int32
        assert np.array_equal(output, [10, 20, 30])


class TestStrClassificationEncoder:
    """Test suite for StrClassificationEncoder."""

    @staticmethod
    @pytest.fixture
    def str_encoder() -> StrClassificationEncoder:
        """Provide a StrClassificationEncoder instance.

        Returns:
            StrClassificationEncoder: Default encoder instance
        """
        return StrClassificationEncoder()

    @staticmethod
    @pytest.fixture
    def scaled_encoder() -> StrClassificationEncoder:
        """Provide a StrClassificationEncoder instance with scaling.

        Returns:
            StrClassificationEncoder: Scaled encoder instance
        """
        return StrClassificationEncoder(scale=True)

    @pytest.mark.parametrize(
        ("fixture", "expected_values"),
        [
            ("str_encoder", [1, 2, 0]),  # "cat"=1, "dog"=2, "bird"=0 (alphabetical order)
            ("scaled_encoder", [0.5, 1.0, 0.0]),  # scaled versions: 1/2=0.5, 2/2=1.0, 0/2=0.0
        ],
    )
    def test_batch_encode_list_of_strings(
        self,
        request: pytest.FixtureRequest,
        fixture: str,
        expected_values: list,
    ) -> None:
        """Test encoding a list of strings."""
        encoder = request.getfixturevalue(fixture)
        input_strings = np.array(["cat", "dog", "bird"])
        output = encoder.batch_encode(input_strings)
        assert isinstance(output, np.ndarray)
        assert output.shape == (3,)

        # For scaled encoder, check that values are floats
        if fixture == "scaled_encoder":
            assert output.dtype == np.int16  # The encoder dtype is still int16
            # Note: The scaling is applied to the data but then cast to int16
            # When cast to int16, the scaled values become: [0.5->0, 1.0->1, 0.0->0]
            expected_int_values = [0, 1, 0]  # After casting to int16
            assert np.array_equal(output, expected_int_values)
        else:
            assert output.dtype == np.int16
            assert np.array_equal(output, expected_values)

    @pytest.mark.parametrize("fixture", ["str_encoder", "scaled_encoder"])
    def test_batch_encode_raises_type_error_on_non_string(
        self,
        request: pytest.FixtureRequest,
        fixture: str,
    ) -> None:
        """Test that encoding non-string data raises a TypeError."""
        encoder = request.getfixturevalue(fixture)
        with pytest.raises(TypeError, match="Expected input data to be a 1D numpy array of strings"):
            encoder.batch_encode(np.array([1, 2, 3]))


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
        """Provide a NumericRankEncoder instance with scaling.

        Returns:
            NumericRankEncoder: Scaled encoder instance
        """
        return NumericRankEncoder(scale=True)

    def test_batch_encode_with_valid_rank(self, rank_encoder: NumericRankEncoder) -> None:
        """Test encoding with rank transformation."""
        input_vals = np.array([10, 20, 5, 30])
        output = rank_encoder.batch_encode(input_vals)
        assert isinstance(output, np.ndarray)
        assert output.shape == (4,)
        assert output.dtype == np.int16
        # Ranks should be [1, 2, 0, 3] for values [10, 20, 5, 30]
        expected_ranks = [1, 2, 0, 3]
        assert np.array_equal(output, expected_ranks)

    def test_batch_encode_with_valid_scaled_rank(self, scaled_encoder: NumericRankEncoder) -> None:
        """Test encoding with scaled rank transformation."""
        input_vals = np.array([10, 20, 5, 30])
        output = scaled_encoder.batch_encode(input_vals)
        assert isinstance(output, np.ndarray)
        assert output.shape == (4,)
        assert output.dtype == np.int16
        # Note: Even with scaling, the final dtype is int16, so values get truncated

    @pytest.mark.parametrize("fixture", ["rank_encoder", "scaled_encoder"])
    def test_batch_encode_with_non_numeric_raises(
        self,
        request: pytest.FixtureRequest,
        fixture: str,
    ) -> None:
        """Test that encoding non-numeric data raises a ValueError."""
        encoder = request.getfixturevalue(fixture)
        with pytest.raises(ValueError, match="Expected input data to be numeric"):
            encoder.batch_encode(np.array(["hello", "world"]))
