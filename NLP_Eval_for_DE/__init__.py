# Define the __all__ variable
__all__ = ["helpers", "driver"]

# Import the submodules
from . import helpers
from . import driver

# Define a variable called version
version = "1.0.0"

# Print a welcome message
print(f"Welcome to NLP eval for DE, version {version}")

