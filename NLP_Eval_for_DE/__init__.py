# Define the __all__ variable
__all__ = ["data", "scores", "results"]

# Import the submodules
from . import data
from . import scores
from . import results

# Define a variable called version
version = "1.0.0"

# Print a welcome message
print(f"Welcome to NLP eval for DE, version {version}")

