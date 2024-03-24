import pytest
import pandas as pd
from badges_count_score import count_documentation_detail, count_correlation_badges_categorical, count_correlation_badges
from unittest.mock import MagicMock, patch

# when count_documentation_detail calls read() on the file object, 
# it will get the content specified 
# in the test case, allowing you to test the function without actually reading from a file.
@pytest.mark.parametrize("file_content, expected_result", [
    ("variable definitions author timestamp", pytest.approx(0.57, 0.01)),
])
def test_count_documentation_detail(file_content, expected_result):
    mock_file = MagicMock()
    mock_file.read.return_value = file_content.encode("utf-8")

    with patch("builtins.open", return_value=mock_file):
        result = count_documentation_detail(mock_file)
        assert result == expected_result



@pytest.mark.parametrize("data, expected_result", [
    (pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}), 1),
    (pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['x', 'y', 'z']}), 0),
    (pd.DataFrame({'A': ['a', 'a', 'b', 'b'], 'B': ['x', 'y', 'x', 'y']}), 0),
])
def test_count_correlation_badges_categorical(data, expected_result):
    result = count_correlation_badges_categorical(data)
    assert result == expected_result


def test_count_correlation_badges():
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 2, 3], 'C': [1, 2, 3]})
    result = count_correlation_badges(data)
    assert result == 1




if __name__ == "__main__":
    pytest.main()
