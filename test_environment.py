import sys

REQUIRED_PYTHON = 3
REQUIRED_PYTHON_MINOR = 8


def main():
    system_major = sys.version_info.major
    system_minor = sys.version_info.minor
    if system_major != REQUIRED_PYTHON:
        raise TypeError(
            "This project requires Python {}. Found: Python {}".format(
                REQUIRED_PYTHON, sys.version))
    else:
        if system_minor < REQUIRED_PYTHON_MINOR:
            raise TypeError(
                "This project requires at least Python {}.{}. Found: Python {}.{}".format(
                    REQUIRED_PYTHON, REQUIRED_PYTHON_MINOR, system_major, system_minor))
        else:
            print(">>> Development environment passes all tests!")


if __name__ == '__main__':
    main()