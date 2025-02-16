# ComfyUI-Transformers-Pipeline - Explicit Node Registration

from .config import be_quiet  # Import the global config - fix this later

# Import all nodes manually since I couldn't get the names to map correctly
from .nodes.batch_processor_tpl import BatchProcessorTpl
from .nodes.caption_export_tpl import CaptionExportTpl
from .nodes.caption_generator_tpl import CaptionGeneratorTpl
from .nodes.florence2_node_tpl import Florence2NodeTpl
from .nodes.image_loader_tpl import ImageLoaderTpl
from .nodes.model_loader_tpl import ModelLoaderTpl
from .nodes.preset_model_list_tpl import PresetModelListTpl
from .nodes.task_list_tpl import TaskListTpl

# Import all utility nodes manually since I couldn't get the names to map correctly
from .util_nodes.debug_node_tpl import DebugNodeTpl
from .util_nodes.exif_metadata_extractor_tpl import ExifMetadataExtractorTpl

# Define node class mappings manually
NODE_CLASS_MAPPINGS = {
    "BatchProcessorTpl": BatchProcessorTpl,
    "CaptionExportTpl": CaptionExportTpl,
    "CaptionGeneratorTpl": CaptionGeneratorTpl,
    "Florence2NodeTpl": Florence2NodeTpl,
    "ImageLoaderTpl": ImageLoaderTpl,
    "ModelLoaderTpl": ModelLoaderTpl,
    "PresetModelListTpl": PresetModelListTpl,
    "TaskListTpl": TaskListTpl,
    "DebugNodeTpl": DebugNodeTpl,
    "ExifMetadataExtractorTpl": ExifMetadataExtractorTpl,
}

#  Define node display names manually
NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchProcessorTpl": "Batch Processor",
    "CaptionExportTpl": "Caption Export",
    "CaptionGeneratorTpl": "Caption Generator",
    "Florence2NodeTpl": "Florence-2 Node",
    "ImageLoaderTpl": "Image Loader",
    "ModelLoaderTpl": "Model Loader",
    "PresetModelListTpl": "Preset Model List",
    "TaskListTpl": "Task List",
    "DebugNodeTpl": "Debug Node",
    "ExifMetadataExtractorTpl": "EXIF Metadata Extractor",
}

# Lets expose the mappings to comfyui like a person in a coat selling watches.
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

