from src.thesis.utils import databricks_test_func

def test_func_2():
    print(f"from {__file__} test_funct_2")
    databricks_test_func

if __name__ == "__main__":
    print(f"from {__file__}")
    databricks_test_func()