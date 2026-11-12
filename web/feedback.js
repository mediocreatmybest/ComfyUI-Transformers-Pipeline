import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const feedbackNodes = new Set([
    "ImageMetadataTpl", "RecordMergeTpl",
    "ZeroShotImageClassificationTpl", "ZeroShotTextClassificationTpl",
    "CaptionPreviewTpl", "CaptionRecordsTpl", "DatasetExportTpl", "CaptionExportTpl",
    "ImageLoaderTpl", "ModelLoaderTpl", "ExifMetadataExtractorTpl", "DebugNodeTpl", "DebugModelNodeTpl",
]);

function updateOutputs(outputs) {
    for (const [id, output] of Object.entries(outputs ?? {})) {
        app.graph?.getNodeById(id)?.tplRenderFeedback?.(output);
    }
}

app.registerExtension({
    name: "TransformersPipeline.Feedback",
    nodeCreated(node) {
        if (!feedbackNodes.has(node.comfyClass)) return;
        const container = document.createElement("div");
        Object.assign(container.style, { display: "flex", flexDirection: "column", overflow: "auto", height: "100%" });
        const text = document.createElement("textarea");
        text.className = "comfy-multiline-input";
        text.readOnly = true;
        text.setAttribute("aria-label", "Caption preview and execution feedback");
        text.value = "Queue this node to see results. Preview content is not saved in the workflow.";
        Object.assign(text.style, { width: "100%", minHeight: "100px", flex: "1", boxSizing: "border-box" });
        const gallery = document.createElement("div");
        container.append(text, gallery);
        node.addDOMWidget("tpl_feedback", "customtext", container, {
            serialize: false,
            getMinHeight: () => node.comfyClass === "CaptionPreviewTpl" ? 240 : 100,
        });
        node.tplRenderFeedback = (message) => {
            text.value = (message?.text ?? []).join("\n").slice(0, 24000);
            gallery.replaceChildren();
            for (const item of (message?.tpl_images ?? []).slice(0, 8)) {
                const figure = document.createElement("figure");
                const image = document.createElement("img");
                const caption = document.createElement("figcaption");
                image.src = api.apiURL("/view?" + new URLSearchParams({
                    filename: item.filename, subfolder: item.subfolder, type: "temp",
                }));
                image.alt = item.label;
                image.loading = "lazy";
                Object.assign(image.style, { maxWidth: "100%", maxHeight: "256px", objectFit: "contain" });
                caption.textContent = item.label;
                figure.append(image, caption);
                gallery.append(figure);
            }
        };
    },
    onNodeOutputsUpdated: updateOutputs,
    afterConfigureGraph() {
        updateOutputs(app.nodeOutputs);
    },
});
