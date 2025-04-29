# Define the __all__ variable
__all__ = ["process_data", "helpers", "results"]

# Import the submodules
from . import process_data
from . import helpers
from . import results

# Define a variable called version
version = "1.0.0"

# Print a welcome message
print(f"Welcome to NLP eval for DE, version {version}")

