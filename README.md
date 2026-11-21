# ComfyUI Transformers Pipeline

## Overview
A collection of additional ComfyUI nodes for utilizing the Hugging Face Transformers pipeline in a modular and efficient manner.

## Installation

Can be installed through the ComfyUI manager or ComfyUI cli

## Example workflows

Open the editor JSON in ComfyUI. The matching files under [`example_workflows/api`](example_workflows/api) are executable API prompts. The examples use relative paths from the ComfyUI working directory. Set the image folder or saved-record path for your own data.

### Getting Started

| Workflow | What it does |
| --- | --- |
| [Caption My Images](example_workflows/Caption_My_Images.json) | Choose an **Image Folder**, generate a short caption per image with BLIP, and preview the results. Turn on **Save Output** when you want files. |
| [Caption + Image Details](example_workflows/Caption_Image_Details.json) | Preview captions and available EXIF together. To include suggested labels, connect a classification branch that uses the same image paths to `records_c` on **Merge / Load Records**. The metadata and labels example below shows those connections. |
| [Prepare My Dataset](example_workflows/Prepare_My_Dataset.json) | Point **Open Saved Records** at a native JSONL file. Review the complete input and optionally use saved classification scores to filter it. **Save Captions** receives the **Kept** output. Connect the loader directly to Save to export all records. This workflow runs without models. |

### Building Blocks

| Workflow | What it teaches |
| --- | --- |
| [Caption Profiles](example_workflows/Caption_Profiles.json) | Generate and compare two Florence caption styles. Optionally save a Full Records archive to reuse in Dataset Export. |
| [Zero-Shot Image Classification](example_workflows/Zero_Shot_Classification.json) | Suggest labels and inspect the original model scores. |
| [Metadata and Labels](example_workflows/Metadata_Merge.json) | Merge EXIF, captions and zero-shot labels by image identity. Optionally save a Full Records archive for the composition and text examples. |
| [Caption Composition](example_workflows/Caption_Composition.json) | Open an archive, append a caption from a template, and preview the original and composed styles together. Adjust **Evidence Options** to select labels by score. |
| [Caption Filter](example_workflows/Caption_Filter.json) | Route records into **Kept**, **Left Out** and **Needs a Look** using saved model scores. Optionally export the kept records as ImageFolder. |
| [Dataset Export](example_workflows/Dataset_Export.json) | Load the archive from Caption Profiles, choose a style, and export the selected captions and images as ImageFolder. |

### Advanced

| Workflow | What it teaches |
| --- | --- |
| [Quantised Captioning](example_workflows/Quantised_Captioning.json) | Configure INT8 and offload placement, then preview or save captions. Requires a compatible environment. |
| [Optional LoRA](example_workflows/Optional_LoRA.json) | Configure a compatible adapter for the chosen base model. |
| [Text Synthesis](example_workflows/Text_Synthesis.json) | Append text-model drafts to saved records and review them alongside original captions. |
| [Zero-Shot Text Classification](example_workflows/Zero_Shot_Text.json) | Classify saved captions and retain links to the source candidates. |
| [Multi-Model Analysis](example_workflows/Analysis_Dataset.json) | Combine captions, two groups of image classifications and EXIF before composition. Optionally save the complete history for re-export. |
| [Analysis Re-export](example_workflows/Analysis_Reexport.json) | Re-export an archive without models using the legacy Dataset Export node, including its profile-filtered native archive. Running this example writes to its configured output folders. |
| [Simple Legacy Sidecars](example_workflows/Simple.json) | Demonstrate the legacy generator and text sidecars. Running this example writes beside the source images. |
| [Florence Captioning](example_workflows/Florence_Caption.json) | Demonstrate the dedicated legacy Florence node and its task controls. Running this example writes beside the source images. |

## Preview and Save Captions

**Caption Records** connects to Preview, Save Captions, or optional processing nodes such as **Merge / Load Records**, **Compose Caption** and **Caption Filter**. These connections carry the records in memory. Use **Full Records (JSONL)** when you want to keep them for later.

**Caption / Image Preview** places each image beside its captions, tags, suggested labels, available camera details and any relevant warnings. Use **Show Details** for candidate IDs, scores and provenance. Set **Page Offset** and **Images per Page** to review more images. Live paging can reuse cached upstream results. Open a saved JSONL directly in Preview for model-free review even after restarting ComfyUI.

In **Save Captions**, connect records or fill in **Saved Records File**. Choose a format and location, then enable **Save Output** and queue the workflow. Leave **Caption Style** blank for records with one candidate per image. For multiple candidates, enter the profile ID shown in Preview details. When multiple candidates share one style, specify **Candidate IDs** in the advanced controls. Training exports skip failed, rejected and unsafe derived captions. Full Records preserves the entire input history.

| File Format | Files | Location and copies |
| --- | --- | --- |
| **Text Captions (TXT)** | One UTF-8 text file per selected image | Use a dedicated folder with optional image copies, or save sidecars beside originals. Sidecars retain the image extension, for example `photo.jpg.txt` and `photo.png.txt`. |
| **Images + Captions (ImageFolder)** | Matching source copies and `metadata.jsonl` | Requires a dedicated folder and **Copy Images**. |
| **CSV** | `captions.csv` with source filename, selected caption, exported image name when copied, profile, review status and tags when present | Use a dedicated folder with optional image copies. The UTF-8 CSV preserves quotes, commas and line breaks. Formula-like cells get a leading apostrophe for spreadsheet safety. |
| **Full Records (JSONL)** | `records.jsonl` with complete candidate, review, analysis and provenance history | Use a dedicated folder with optional image copies. Saved records keep their source paths. Copies are additional files for convenience. |

**Keep Existing Files** protects the batch if it would reuse existing output. **Create New Version** allocates a versioned folder or uses one shared suffix for the whole sidecar batch. **Replace Existing Output** is explicit. Use a new folder when the output file set changes. Keep copies outside the recursive input folder to avoid discovering them on a later run. Turn **Save Output** off again before requeueing the full workflow for preview, or queue only the Preview output if your ComfyUI version supports it.

Saved workflows can still use **Caption Generator**, **Caption Export**, and **Dataset Export** with their existing sockets and controls.

## ComfyUI Desktop acceptance

1. Update the extension in Desktop's active `custom_nodes` directory to `feature/ux-first-impression`, restart Desktop and refresh the frontend. Open `Caption_My_Images.json`. In **Image Folder**, choose a disposable folder with several visually distinct JPEG or PNG images, including a pair like `photo.jpg` and `photo.png`. Leave the BLIP model at its saved value. Queue the workflow. An uncached model may download on Desktop.
2. With **Save Output** off, check that each thumbnail, filename and caption match and that no caption files or export folder appeared. Set **Images per Page** to 1 and queue at **Page Offset** 0 and 1. Check alignment and that cached captions are reused while the input files and model settings stay unchanged. Expand **Show Details** and confirm technical fields are collapsed initially.
3. Turn **Save Output** on. Choose **Text Captions (TXT)**, **Save to a folder I choose**, **Copy Images** off and a fresh output folder. Queue and verify one distinct TXT per successful image and the saved count.
4. Set **Copy Images** on and choose a second fresh output folder. Queue and verify each copied image has a same-stem TXT. Then select **Images + Captions (ImageFolder)** in a third fresh folder and verify every `file_name` in `metadata.jsonl` has a matching copy.
5. Choose **Text Captions (TXT)**, **Save captions beside the original images** and **Copy Images** off. Queue and verify `photo.jpg.txt` and `photo.png.txt` are distinct. Check that original image contents are unchanged.
6. Select **CSV**, return to **Save to a folder I choose**, set **Copy Images** off and choose a fresh folder. Queue and open `captions.csv` as UTF-8. Check filenames, captions and row counts against Preview.
7. Select **Full Records (JSONL)** and a fresh folder. Queue and check `records.jsonl`, then open it in **Prepare My Dataset**. Queue with saving off and verify captions and thumbnails appear without a model loader.
8. Open **Caption Composition**. Point **Open Saved Records** at that `records.jsonl`, set the template to `{caption} Desktop edit.` and queue. Check that Preview shows both styles. In **Save Captions**, set **Caption Style** to `composed`, choose a fresh TXT output folder and turn **Save Output** on. Queue and check the updated text. Change the template again and use explicit **Replace Existing Output** to verify the new text replaces the prior export.
9. Switch **Existing Files** back to **Keep Existing Files** and requeue to the same destination. Expect a protected message with zero saved and verify the existing files are intact. Try **Create New Version** and check that the batch goes into a single new folder.
10. Open the other starter workflows and the advanced options in the table. Confirm that nodes and links load and groups are readable at normal zoom. Run **Caption + Image Details** with one photo with EXIF and one without. Check the camera line appears only where data exists. Test advanced model execution only with the relevant Desktop dependencies and hardware.

## TODO

### **Short Term**
- [ ] Add tooltips nodes.
- [ ] Move pipeline nodes into its own folders (that’s our primary focus really) with non-pipeline tasks separated.
- [ ] fix logging to possibly Python’s logging module. 
- [ ] Add additional example nodes

### **Long Term**

- [x] Register in ComfyUI registry
- [ ] look into multi-pass captions or outputs
- [ ] add additional pipeline tasks
- [ ] add multimodal tasks to the mix
- [ ] Add additional output nodes
- [ ] Allow setting different file extensions for output

## Contributing
Pull requests are welcome.
As well as Ideas, suggestions, anything at all. 
What works, what doesn’t, etc.

## License
Please see license file. 

