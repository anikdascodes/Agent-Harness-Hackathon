import pytest
import os
import sys

# Add project root and sandbox-env to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(root_dir, "sandbox-env"))
from executor import execute_code, validate_code_safety

def test_basic_execution():
    code = """
x = 10
y = 20
print(f"SUM={x + y}")
"""
    res = execute_code(code)
    assert res["success"] is True
    assert "SUM=30" in res["stdout"]
    assert res["error"] is None

def test_syntax_error_handling():
    code = "def broken(:"
    res = execute_code(code)
    assert res["success"] is False
    assert res["error"] is not None

def test_security_ast_validation():
    # Attempting to import restricted module should fail before execution
    malicious_code = "import subprocess\nsubprocess.run(['ls'])"
    is_safe, err = validate_code_safety(malicious_code)
    assert is_safe is False
    assert "Security Violation" in err

    res = execute_code(malicious_code)
    assert res["success"] is False
    assert "Security Violation" in res["stderr"]

def test_plot_capture():
    code = """
import matplotlib.pyplot as plt
plt.figure()
plt.plot([1, 2, 3], [4, 5, 6])
plt.title("Test Plot")
plt.show()
"""
    res = execute_code(code)
    assert res["success"] is True
    assert len(res["plots"]) >= 1
    assert os.path.exists(res["plots"][0]["absolute_path"])
