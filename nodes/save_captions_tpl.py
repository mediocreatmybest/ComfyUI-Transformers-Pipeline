import json
from pathlib import Path

from ..captioning.export import export_records, load_export_records
from .ui_feedback import feedback


FORMATS = {
    'Text Captions (TXT)': 'txt',
    'Images + Captions (ImageFolder)': 'imagefolder',
    'CSV': 'csv',
    'Full Records (JSONL)': 'native',
}
FOLDER = 'Save to a folder I choose'
BESIDE = 'Save captions beside the original images'
KEEP = 'Keep Existing Files'
VERSION = 'Create New Version'
REPLACE = 'Replace Existing Output'


class SaveCaptionsTpl:
    DESCRIPTION = 'Save connected captions or export a saved JSONL file without running a model. Turn on Save Output when ready. Full Records archives every supplied candidate and analysis.'

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'save_output': ('BOOLEAN', {'default': False, 'tooltip': 'Enable writing when you queue the workflow.'}),
            'file_format': (list(FORMATS), {'tooltip': 'TXT sidecars or a separate folder with TXT, ImageFolder, CSV, or full native JSONL records.'}),
            'save_location': ([FOLDER, BESIDE], {'tooltip': 'Saving beside the original images is available for TXT only.'}),
            'destination': ('STRING', {'default': './output/caption_exports', 'tooltip': 'Choose a dedicated export folder. Relative paths start from the ComfyUI working directory.'}),
            'copy_images': ('BOOLEAN', {'default': False, 'tooltip': 'Include source image copies in a separate folder. Turn this on for ImageFolder.'}),
            'file_policy': ([KEEP, VERSION, REPLACE], {'tooltip': 'Keep cancels the batch if output exists. New Version gives the whole batch a new destination or sidecar suffix. Replace updates matching output files.'}),
        }, 'optional': {
            'caption_records': ('LIST', {'tooltip': 'Connect Caption Records or a processing node.'}),
            'records_file': ('STRING', {'default': '', 'tooltip': 'Use a saved native JSONL file instead of the connection.'}),
            'profile_id': ('STRING', {'default': '', 'tooltip': 'Caption style/profile ID. Required when a record has multiple matching candidates. Full Records archives all styles.'}),
            'caption_type': ('STRING', {'default': '', 'advanced': True, 'tooltip': 'Optional filter by caption type.'}),
            'candidate_ids': ('STRING', {'default': '{}', 'advanced': True, 'tooltip': 'Optional JSON mapping of image IDs to candidate IDs when style alone is ambiguous.'}),
            'approved_only': ('BOOLEAN', {'default': False, 'advanced': True, 'tooltip': 'Only export reviewed and approved training captions. Full Records preserves all review states.'}),
        }}

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('status',)
    FUNCTION = 'execute'
    OUTPUT_NODE = True
    CATEGORY = 'Transformers Pipeline/Datasets'

    @classmethod
    def IS_CHANGED(cls, save_output=False, **kwargs):
        # Requeueing an enabled save must recheck the filesystem each time.
        return False if save_output is False else float('nan')

    def execute(self, save_output=False, file_format='Text Captions (TXT)', save_location=FOLDER,
                destination='./output/caption_exports', copy_images=False, file_policy=KEEP,
                caption_records=None, records_file='', profile_id='', caption_type='', candidate_ids='{}',
                approved_only=False):
        location = 'Beside original images' if save_location == BESIDE else str(Path(destination).expanduser())
        heading = f'Format: {file_format}\nDestination: {location}'
        if not save_output:
            status = f'Save Output is off.\n{heading}'
            return feedback((status,), status)
        records = load_export_records(caption_records, records_file)
        try:
            report = export_records(records, destination, FORMATS[file_format], profile_id=profile_id.strip(),
                caption_type=caption_type.strip(), candidate_ids=json.loads(candidate_ids) if FORMATS[file_format] != 'native' else None,
                copy_images=copy_images, beside=save_location == BESIDE, archive_all=True,
                overwrite=file_policy == REPLACE, version=file_policy == VERSION, approved_only=approved_only,
                allow_empty=True, input_file=records_file)
        except FileExistsError as exc:
            status = (f'Saved 0, Skipped {len(records)}.\n{heading}\nProtected existing output. {exc}\n'
                      'Choose Create New Version or select another Output Folder.')
        else:
            reasons = '\n'.join(f"{item['source_filename']}: {item['reason']}"[:400] for item in report['skipped'][:8])
            unit = 'records' if FORMATS[file_format] == 'native' else 'captions'
            status = (f"Saved {report['exported']} {unit}, Skipped {len(report['skipped'])}.\n"
                      f"Format: {file_format}\nDestination: {report['destination']}")
            if report.get('scope'):
                status += '\n' + report['scope']
            if report['files'] and file_policy == VERSION and save_location == BESIDE:
                status += '\nExample sidecar: ' + report['files'][0]
            if reasons:
                status += '\n' + reasons
            elif not records:
                status += '\nNo records received. Check the upstream image or filter selection.'
        return feedback((status,), status)
