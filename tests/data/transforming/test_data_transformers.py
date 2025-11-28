"""Test suite for the data transformation generators."""

from typing import Any

import numpy as np
import pytest

from src.stimulus.data.transforming.transforms import (
    AbstractTransform,
    BalanceSampler,
    GaussianChunk,
    GaussianNoise,
    RandomDownSampler,
    ReverseComplement,
    SwapTransform,
    UniformTextMasker,
)


class DataTransformerTest:
    """Data for a data transformer test.

    Attributes:
        transformer: The data transformer to test.
        params: The parameters to use for the test.
        single_input: The single input to transform.
        expected_single_output: The expected output for the single input.
        multiple_inputs: The multiple inputs to transform.
        expected_multiple_outputs: The expected outputs for the multiple inputs.
    """

    def __init__(  # noqa: D107
        self,
        transformer: AbstractTransform,
        params: dict,
        single_input: Any,
        expected_single_output: Any,
        multiple_inputs: Any,
        expected_multiple_outputs: Any,
    ):
        self.transformer = transformer
        self.params = params
        self.single_input = single_input
        self.expected_single_output = expected_single_output
        self.multiple_inputs = multiple_inputs
        self.expected_multiple_outputs = expected_multiple_outputs


@pytest.fixture
def uniform_text_masker() -> DataTransformerTest:
    """Return a UniformTextMasker test object."""
    np.random.seed(42)  # Set seed before creating transformer
    transformer = UniformTextMasker(mask="N", probability=0.1)
    params: dict[str, Any] = {}  # Remove seed from params
    single_input = "ACGTACGT"
    expected_single_output = "ACGTACNT"
    multiple_inputs = ["ATCGATCGATCG", "ATCG"]
    expected_multiple_outputs = ["ATCGATNGATNG", "ATCG"]
    return DataTransformerTest(
        transformer=transformer,
        params=params,  # Empty params dict since seed is handled during initialization
        single_input=single_input,
        expected_single_output=expected_single_output,
        multiple_inputs=multiple_inputs,
        expected_multiple_outputs=expected_multiple_outputs,
    )


@pytest.fixture
def gaussian_noise() -> DataTransformerTest:
    """Return a GaussianNoise test object."""
    np.random.seed(42)  # Set seed before creating transformer
    transformer = GaussianNoise(mean=0, std=1)
    params: dict[str, Any] = {}  # Remove seed from params
    single_input = 5.0
    expected_single_output = 5.4967141530112327
    multiple_inputs = [1.0, 2.0, 3.0]
    expected_multiple_outputs = [
        1.4967141530112327,
        1.8617356988288154,
        3.6476885381006925,
    ]
    return DataTransformerTest(
        transformer=transformer,
        params=params,
        single_input=single_input,
        expected_single_output=expected_single_output,
        multiple_inputs=multiple_inputs,
        expected_multiple_outputs=expected_multiple_outputs,
    )


@pytest.fixture
def gaussian_chunk() -> DataTransformerTest:
    """Return a GaussianChunk test object."""
    np.random.seed(42)  # Set seed before creating transformer
    transformer = GaussianChunk(chunk_size=2)
    params: dict[str, Any] = {}  # Remove seed from params
    single_input = "ACGT"
    expected_single_output = "CG"
    multiple_inputs = ["ACGT", "TGCA"]
    expected_multiple_outputs = ["CG", "GC"]
    return DataTransformerTest(
        transformer=transformer,
        params=params,
        single_input=single_input,
        expected_single_output=expected_single_output,
        multiple_inputs=multiple_inputs,
        expected_multiple_outputs=expected_multiple_outputs,
    )


@pytest.fixture
def reverse_complement() -> DataTransformerTest:
    """Return a ReverseComplement test object."""
    transformer = ReverseComplement()
    single_input = "ACCCCTACGTNN"
    expected_single_output = "NNACGTAGGGGT"
    multiple_inputs = ["ACCCCTACGTNN", "ACTGA"]
    expected_multiple_outputs = ["NNACGTAGGGGT", "TCAGT"]
    return DataTransformerTest(
        transformer=transformer,
        params={},
        single_input=single_input,
        expected_single_output=expected_single_output,
        multiple_inputs=multiple_inputs,
        expected_multiple_outputs=expected_multiple_outputs,
    )


class TestUniformTextMasker:
    """Test suite for the UniformTextMasker class."""

    @pytest.mark.parametrize("test_data_name", ["uniform_text_masker"])
    def test_transform_single(self, request: Any, test_data_name: str) -> None:
        """Test masking a single string.

        Args:
            test_data: The test data to use.
        """
        test_data = request.getfixturevalue(test_data_name)
        transformed_data = test_data.transformer.transform(
            test_data.single_input,
            **test_data.params,
        )
        assert isinstance(transformed_data, str)
        assert transformed_data == test_data.expected_single_output

    @pytest.mark.parametrize("test_data_name", ["uniform_text_masker"])
    def test_transform_multiple(self, request: Any, test_data_name: str) -> None:
        """Test masking multiple strings."""
        test_data = request.getfixturevalue(test_data_name)
        transformed_data = [test_data.transformer.transform(x, **test_data.params) for x in test_data.multiple_inputs]
        assert isinstance(transformed_data, list)
        for item in transformed_data:
            assert isinstance(item, str)
        assert transformed_data == test_data.expected_multiple_outputs


class TestGaussianNoise:
    """Test suite for the GaussianNoise class."""

    @pytest.mark.parametrize("test_data_name", ["gaussian_noise"])
    def test_transform_single(self, request: Any, test_data_name: str) -> None:
        """Test transforming a single float."""
        test_data = request.getfixturevalue(test_data_name)
        transformed_data = test_data.transformer.transform(
            test_data.single_input,
            **test_data.params,
        )
        assert isinstance(transformed_data, float)
        assert round(transformed_data, 7) == round(test_data.expected_single_output, 7)

    @pytest.mark.parametrize("test_data_name", ["gaussian_noise"])
    def test_transform_multiple(
        self,
        request: Any,
        test_data_name: DataTransformerTest,
    ) -> None:
        """Test transforming multiple floats."""
        test_data = request.getfixturevalue(test_data_name)
        transformed_data = test_data.transformer.transform_all(
            test_data.multiple_inputs,
            **test_data.params,
        )
        assert isinstance(transformed_data, list)
        for item in transformed_data:
            assert isinstance(item, float)
        assert len(transformed_data) == len(test_data.expected_multiple_outputs)
        for item, expected in zip(
            transformed_data,
            test_data.expected_multiple_outputs,
        ):
            assert round(item, 7) == round(expected, 7)


class TestGaussianChunk:
    """Test suite for the GaussianChunk class."""

    @pytest.mark.parametrize("test_data_name", ["gaussian_chunk"])
    def test_transform_single(self, request: Any, test_data_name: str) -> None:
        """Test transforming a single string."""
        test_data = request.getfixturevalue(test_data_name)
        transformed_data = test_data.transformer.transform(test_data.single_input)
        assert isinstance(transformed_data, str)
        assert len(transformed_data) == 2

    @pytest.mark.parametrize("test_data_name", ["gaussian_chunk"])
    def test_transform_multiple(self, request: Any, test_data_name: str) -> None:
        """Test transforming multiple strings."""
        test_data = request.getfixturevalue(test_data_name)
        transformed_data = [test_data.transformer.transform(x) for x in test_data.multiple_inputs]
        assert isinstance(transformed_data, list)
        for item in transformed_data:
            assert isinstance(item, str)
            assert len(item) == 2
        assert transformed_data == test_data.expected_multiple_outputs

    @pytest.mark.parametrize("test_data_name", ["gaussian_chunk"])
    def test_chunk_size_excessive(self, request: Any, test_data_name: str) -> None:
        """Test that the transform fails if chunk size is greater than the length of the input string."""
        test_data = request.getfixturevalue(test_data_name)
        transformer = GaussianChunk(chunk_size=100)
        with pytest.raises(
            ValueError,
            match="The input data is shorter than the chunk size",
        ):
            transformer.transform(test_data.single_input)


class TestReverseComplement:
    """Test suite for the ReverseComplement class."""

    @pytest.mark.parametrize("test_data_name", ["reverse_complement"])
    def test_transform_single(self, request: Any, test_data_name: str) -> None:
        """Test transforming a single string."""
        test_data = request.getfixturevalue(test_data_name)
        transformed_data = test_data.transformer.transform(
            test_data.single_input,
            **test_data.params,
        )
        assert isinstance(transformed_data, str)
        assert transformed_data == test_data.expected_single_output

    @pytest.mark.parametrize("test_data_name", ["reverse_complement"])
    def test_transform_multiple(self, request: Any, test_data_name: str) -> None:
        """Test transforming multiple strings."""
        test_data = request.getfixturevalue(test_data_name)
        transformed_data = test_data.transformer.transform_all(
            test_data.multiple_inputs,
            **test_data.params,
        )
        assert isinstance(transformed_data, list)
        for item in transformed_data:
            assert isinstance(item, str)
        assert transformed_data == test_data.expected_multiple_outputs


def test_balance_sampler() -> None:
    """Test the BalanceSampler class."""
    sampler = BalanceSampler()
    data = ["a", "a", "a", "b", "b", "c", "c", "c", "c", "c"]
    transformed_data = sampler.transform_all(data)
    nb_a = len([x for x in transformed_data if x == "a"])
    nb_b = len([x for x in transformed_data if x == "b"])
    nb_c = len([x for x in transformed_data if x == "c"])
    nb_nan = len([x for x in transformed_data if isinstance(x, float) and np.isnan(x)])
    assert nb_a == nb_b == nb_c == 2
    assert nb_nan == 4


def test_swap_transform() -> None:
    """Test the SwapTransform class."""
    swap_transform = SwapTransform(swap_numbers=1.5)
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    transformed_data = swap_transform.transform_all(data)
    assert transformed_data != data
    # check that only two elements are swapped
    # Count differences between original and transformed data
    differences = sum(1 for i, x in enumerate(transformed_data) if x != data[i])
    assert differences == 2


def test_swap_transform_multiple() -> None:
    """Test the SwapTransform class with multiple swaps."""
    swap_transform = SwapTransform(swap_numbers=10)
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    transformed_data = swap_transform.transform_all(data)
    assert transformed_data != data


def test_random_down_sampler() -> None:
    """Test the RandomDownSampler class."""
    sampler = RandomDownSampler(n=3, seed=0)
    data = ["a", "a", "a", "b", "b", "c", "c", "c", "c", "c"]
    transformed_data = sampler.transform_all(data)
    transformed_data = [x for x in transformed_data if not isinstance(x, float) or not np.isnan(x)]
    assert len(transformed_data) == 3
    assert transformed_data[0] == "a"
    assert transformed_data[1] == "b"
    assert transformed_data[2] == "c"
