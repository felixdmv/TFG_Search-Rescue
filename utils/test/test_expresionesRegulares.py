import pytest
from utils.expresionesRegulares import getPatronFile


def test_getPatronFile():
    """
    Test function for the getPatronFile function.

    This function tests the behavior of the getPatronFile function by asserting the expected output for different test cases.

    Test Cases:
    - Caso 1: Coincidencia simple
    - Caso 2: Coincidencia múltiple
    - Caso 3: Sin coincidencia
    - Caso 4: Coincidencia con caracteres especiales
    - Caso 5: Patrón en el medio
    - Caso 6: Patrón con delimitadores
    """
    # Caso 1: Coincidencia simple
    base_name = "file123_test"
    regex = r"file(\d+)_test"
    position = 1
    assert getPatronFile(base_name, regex, position) == "123"

    # Caso 2: Coincidencia múltiple
    base_name = "abc123_def456"
    regex = r"abc(\d+)_def(\d+)"
    position = 2
    assert getPatronFile(base_name, regex, position) == "456"

    # Caso 3: Sin coincidencia
    base_name = "no_match_here"
    regex = r"file(\d+)_test"
    position = 1
    assert getPatronFile(base_name, regex, position) is None

    # Caso 4: Coincidencia con caracteres especiales
    base_name = "example_file_2023_06_28"
    regex = r"example_file_(\d{4})_(\d{2})_(\d{2})"
    position = 1
    assert getPatronFile(base_name, regex, position) == "2023"
    position = 2
    assert getPatronFile(base_name, regex, position) == "06"
    position = 3
    assert getPatronFile(base_name, regex, position) == "28"

    # Caso 5: Patrón en el medio
    base_name = "start_middle_end"
    regex = r"start_(middle)_end"
    position = 1
    assert getPatronFile(base_name, regex, position) == "middle"

    # Caso 6: Patrón con delimitadores
    base_name = "user_42_email"
    regex = r"user_(\d+)_email"
    position = 1
    assert getPatronFile(base_name, regex, position) == "42"


if __name__ == "__main__":
    pytest.main()
