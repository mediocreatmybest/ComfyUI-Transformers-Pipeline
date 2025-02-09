import os
import importlib
from .config import be_quiet  # Import global config


# Lets automagic import all node files inside the main "nodes" directory
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Specify the nodes directory, need to add utils in future.
nodes_dir = os.path.dirname(__file__) + "/nodes"

# Load all nodes from the nodes directory and add them to the mappings
for filename in os.listdir(nodes_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = filename[:-3]  # Remove .py extension
        try:
            module = importlib.import_module(f".nodes.{module_name}", package=__name__)
            class_name = "".join(word.capitalize() for word in module_name.split("_"))  # Convert to PascalCase
            if hasattr(module, class_name):
                NODE_CLASS_MAPPINGS[class_name] = getattr(module, class_name)
                NODE_DISPLAY_NAME_MAPPINGS[class_name] = class_name.replace("_", " ")
        # Throw an error if the modules can't be loaded
        except Exception as e:
            print(f"Error loading {module_name}: {e}")

# Lets expose the mappings to comfyui like a person in a coat selling watches.
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']


