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
    ModelLoaderTpl: { model_preset: "Model Preset", model_name_or_path: "Custom Model ID or Path", task: "Custom Task" },
    CaptionRecordsTpl: { caption_style: "Caption Style", profile_ids: "Custom Profile IDs",
        profiles_json: "Profiles and Prompts (JSON)", prompt_override: "Prompt Override" },
    RecordMergeTpl: { records_file: "Saved Records File" },
    CaptionPreviewTpl: { records_file: "Saved Records File", profile_id: "Caption Style", offset: "Page Offset", limit: "Images per Page" },
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
    const row = document.createElement("div");
    Object.assign(row.style, { display: "flex", alignItems: "flex-start", gap: "8px" });
    if (card.image) {
        const image = thumbnail(card.image);
        Object.assign(image.style, { maxWidth: "30%", maxHeight: "112px", height: "auto" });
        row.append(image);
    }
    const captions = document.createElement("div");
    captions.style.minWidth = "0";
    for (const candidate of card.captions ?? []) {
        const block = document.createElement("div");
        const caption = textElement("div", candidate.text);
        caption.style.whiteSpace = "pre-wrap";
        block.append(textElement("small", candidate.label), caption);
        if (candidate.tags?.length) block.append(textElement("small", "Tags · " + candidate.tags.join(", ")));
        captions.append(block);
    }
    row.append(captions);
    article.append(row);
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

function updateOutputs(outputs, ids = Object.keys(outputs ?? {})) {
    for (const id of ids) {
        const output = outputs?.[id];
        if (output) app.graph?.getNodeById(id)?.tplRenderFeedback?.(output);
    }
}

app.registerExtension({
    name: "TransformersPipeline.Feedback",
    setup() {
        api.addEventListener("execution_cached", ({ detail }) => {
            updateOutputs(app.nodeOutputs, detail.nodes);
        });
    },
    nodeCreated(node) {
        if (!feedbackNodes.has(node.comfyClass)) return;
        const labels = widgetLabels[node.comfyClass] ?? {};
        for (const widget of node.widgets ?? []) {
            if (labels[widget.name]) widget.label = labels[widget.name];
        }
        const container = document.createElement("div");
        const isPreview = node.comfyClass === "CaptionPreviewTpl";
        const minHeight = isPreview ? 240 : 100;
        container.tabIndex = 0;
        container.setAttribute("data-capture-wheel", "true");
        container.setAttribute("role", "region");
        container.setAttribute("aria-label", isPreview ? "Caption and image preview" : "Execution feedback");
        Object.assign(container.style, {
            width: "100%", height: "100%", minWidth: "0", minHeight: "0", boxSizing: "border-box",
            // Canvas supplies a fixed box. Auto-sized Vue rows need an intrinsic viewport independent of the cards.
            contain: "size", containIntrinsicSize: `0px ${minHeight}px`, overflow: "auto", overscrollBehavior: "contain",
            color: "var(--input-text, #eee)", backgroundColor: "var(--comfy-input-bg, #252525)",
        });
        const text = document.createElement("textarea");
        text.className = "comfy-multiline-input";
        text.readOnly = true;
        text.setAttribute("aria-label", "Caption preview and execution feedback");
        text.value = "Queue this node to see results. Preview content is not saved in the workflow.";
        Object.assign(text.style, { width: "100%", height: "100%", minHeight: "100px", resize: "none", boxSizing: "border-box" });
        const gallery = document.createElement("div");
        container.append(text, gallery);
        // "customtext" selects ComfyUI's native textarea on older Vue renderers.
        node.addDOMWidget("tpl_feedback", "tpl_feedback", container, {
            serialize: false,
            getMinHeight: () => minHeight,
            hideOnZoom: !isPreview,
        });
        let lastContent;
        node.tplRenderFeedback = (message) => {
            if (!message) return;
            // ComfyUI clones cached outputs. Replaying the same content must keep scroll and open details.
            const content = JSON.stringify([message.text, message.tpl_cards, message.tpl_images, message.tpl_summary]);
            if (content === lastContent) return;
            text.value = (message.text ?? []).join("\n").slice(0, 24000);
            gallery.replaceChildren();
            const hasCards = Array.isArray(message.tpl_cards)
                && (message.tpl_cards.length > 0 || !!message.tpl_summary?.length);
            text.hidden = hasCards;
            container.scrollTop = 0;
            if (hasCards) {
                gallery.append(textElement("p", (message.tpl_summary ?? []).join(" ")));
                for (const card of message.tpl_cards.slice(0, 8)) gallery.append(imageCard(card));
            } else {
                for (const item of (message.tpl_images ?? []).slice(0, 8)) {
                    const figure = document.createElement("figure");
                    figure.append(thumbnail(item), textElement("figcaption", item.label));
                    gallery.append(figure);
                }
            }
            lastContent = content;
            node.setDirtyCanvas?.(true, false);
        };
        // Normal execution delivers a per-node callback. Bulk updates are used for history and restore.
        const onExecuted = node.onExecuted;
        node.onExecuted = function (message, ...args) {
            const result = onExecuted?.call(this, message, ...args);
            this.tplRenderFeedback(message);
            return result;
        };
    },
    onNodeOutputsUpdated(outputs) {
        updateOutputs(outputs);
    },
    afterLoadGraph() {
        updateOutputs(app.nodeOutputs);
    },
    afterConfigureGraph() {
        updateOutputs(app.nodeOutputs);
    },
});
