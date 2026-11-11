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
from .nodes.caption_records_tpl import CaptionRecordsTpl
from .nodes.dataset_export_tpl import DatasetExportTpl
from .nodes.caption_preview_tpl import CaptionPreviewTpl

# Import all utility nodes manually since I couldn't get the names to map correctly
from .util_nodes.debug_node_tpl import DebugNodeTpl
from .util_nodes.exif_metadata_extractor_tpl import ExifMetadataExtractorTpl
from .util_nodes.debug_model_node_tpl import DebugModelNodeTpl

# Define node class mappings manually
NODE_CLASS_MAPPINGS = {
    "CaptionPreviewTpl": CaptionPreviewTpl,
    "CaptionRecordsTpl": CaptionRecordsTpl,
    "DatasetExportTpl": DatasetExportTpl,
    "BatchProcessorTpl": BatchProcessorTpl,
    "CaptionExportTpl": CaptionExportTpl,
    "CaptionGeneratorTpl": CaptionGeneratorTpl,
    "Florence2NodeTpl": Florence2NodeTpl,
    "ImageLoaderTpl": ImageLoaderTpl,
    "ModelLoaderTpl": ModelLoaderTpl,
    "PresetModelListTpl": PresetModelListTpl,
    "TaskListTpl": TaskListTpl,
    "DebugNodeTpl": DebugNodeTpl,
    "DebugModelNodeTpl": DebugModelNodeTpl,
    "ExifMetadataExtractorTpl": ExifMetadataExtractorTpl,
}

#  Define node display names manually
NODE_DISPLAY_NAME_MAPPINGS = {
    "CaptionPreviewTpl": "Caption / Image Preview",
    "CaptionRecordsTpl": "Caption Records",
    "DatasetExportTpl": "Dataset Export",
    "BatchProcessorTpl": "Batch Processor (Path Groups)",
    "CaptionExportTpl": "Caption Export",
    "CaptionGeneratorTpl": "Caption Generator",
    "Florence2NodeTpl": "Florence-2 Node",
    "ImageLoaderTpl": "Image Loader",
    "ModelLoaderTpl": "Model Loader",
    "PresetModelListTpl": "Caption Model List",
    "TaskListTpl": "Task List",
    "DebugNodeTpl": "Debug Node",
    "DebugModelNodeTpl": "Debug Pipeline Node",
    "ExifMetadataExtractorTpl": "EXIF Metadata Extractor",
}

# Lets expose the mappings to comfyui like a person in a coat selling watches.
WEB_DIRECTORY = './web'
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

