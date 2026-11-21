import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const feedbackNodes = new Set([
    "CaptionFilterTpl",
    "CaptionComposeTpl", "CaptionSynthesisTpl",
    "ImageMetadataTpl", "RecordMergeTpl",
    "ZeroShotImageClassificationTpl", "ZeroShotTextClassificationTpl",
    "CaptionPreviewTpl", "CaptionRecordsTpl", "DatasetExportTpl", "CaptionExportTpl", "SaveCaptionsTpl",
    "ImageLoaderTpl", "ModelLoaderTpl", "ExifMetadataExtractorTpl", "DebugNodeTpl", "DebugModelNodeTpl",
]);

const widgetLabels = {
    CaptionRecordsTpl: { profile_ids: "Caption Style" },
    CaptionPreviewTpl: { profile_id: "Caption Style", offset: "Page Offset", limit: "Images per Page" },
    SaveCaptionsTpl: { save_output: "Save Output", file_format: "File Format", save_location: "Save Location",
        destination: "Output Folder", copy_images: "Copy Images", file_policy: "Existing Files",
        profile_id: "Caption Style", records_file: "Saved Records File" },
    CaptionFilterTpl: { group_id: "Which group of labels?", label: "Keep images matching this label",
        minimum_score: "Minimum Score", no_score: "No Score Available" },
};

function textElement(tag, value) {
    const element = document.createElement(tag);
    element.textContent = value;
    return element;
}

function thumbnail(item) {
    const image = document.createElement("img");
    image.src = api.apiURL("/view?" + new URLSearchParams({
        filename: item.filename, subfolder: item.subfolder, type: "temp",
    }));
    image.alt = item.label;
    image.loading = "lazy";
    Object.assign(image.style, { maxWidth: "100%", maxHeight: "256px", objectFit: "contain" });
    return image;
}

function imageCard(card) {
    const article = document.createElement("article");
    Object.assign(article.style, {
        display: "grid", gap: "8px", padding: "12px", marginBottom: "8px",
        border: "1px solid var(--border-color, #666)", borderRadius: "8px", overflowWrap: "anywhere",
    });
    article.append(textElement("strong", card.filename));
    if (card.image) article.append(thumbnail(card.image));
    for (const candidate of card.captions ?? []) {
        const block = document.createElement("div");
        const caption = textElement("div", candidate.text);
        caption.style.whiteSpace = "pre-wrap";
        block.append(textElement("small", candidate.label), caption);
        if (candidate.tags?.length) block.append(textElement("small", "Tags · " + candidate.tags.join(", ")));
        article.append(block);
    }
    for (const label of card.classifications ?? []) article.append(textElement("small", label));
    if (card.camera) article.append(textElement("small", card.camera));
    for (const warning of card.warnings ?? []) article.append(textElement("div", warning));
    if (card.details) {
        const details = document.createElement("details");
        const content = textElement("pre", card.details);
        content.style.whiteSpace = "pre-wrap";
        details.append(textElement("summary", "Show Details"), content);
        article.append(details);
    }
    return article;
}

function updateOutputs(outputs) {
    for (const [id, output] of Object.entries(outputs ?? {})) {
        app.graph?.getNodeById(id)?.tplRenderFeedback?.(output);
    }
}

app.registerExtension({
    name: "TransformersPipeline.Feedback",
    nodeCreated(node) {
        if (!feedbackNodes.has(node.comfyClass)) return;
        const labels = widgetLabels[node.comfyClass] ?? {};
        for (const widget of node.widgets ?? []) {
            if (labels[widget.name]) widget.label = labels[widget.name];
        }
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
            const hasCards = Array.isArray(message?.tpl_cards);
            text.hidden = hasCards;
            if (hasCards) {
                gallery.append(textElement("p", (message.tpl_summary ?? []).join(" ")));
                for (const card of message.tpl_cards.slice(0, 8)) gallery.append(imageCard(card));
                return;
            }
            for (const item of (message?.tpl_images ?? []).slice(0, 8)) {
                const figure = document.createElement("figure");
                figure.append(thumbnail(item), textElement("figcaption", item.label));
                gallery.append(figure);
            }
        };
    },
    onNodeOutputsUpdated: updateOutputs,
    afterConfigureGraph() {
        updateOutputs(app.nodeOutputs);
    },
});
