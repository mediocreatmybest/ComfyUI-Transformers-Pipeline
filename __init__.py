from ctypes import util
import os
import importlib
from .config import be_quiet  # Import global config


# Lets automagic import all node files inside the main "nodes" directory
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Specify the main nodes and util nodes directory.
nodes_dir = os.path.dirname(__file__) + "/nodes"
utils_dir = os.path.dirname(__file__) + "/util_nodes"

# Load all nodes from two directories and add them to the mappings
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

# Quick and lazy way to load the util nodes for now.
for filename in os.listdir(utils_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = filename[:-3]  # Remove .py extension
        try:
            module = importlib.import_module(f".util_nodes.{module_name}", package=__name__)
            class_name = "".join(word.capitalize() for word in module_name.split("_"))  # Convert to PascalCase
            if hasattr(module, class_name):
                NODE_CLASS_MAPPINGS[class_name] = getattr(module, class_name)
                NODE_DISPLAY_NAME_MAPPINGS[class_name] = class_name.replace("_", " ")
        # Throw an error if the modules can't be loaded
        except Exception as e:
            print(f"Error loading {module_name}: {e}")

# Lets expose the mappings to comfyui like a person in a coat selling watches.
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
