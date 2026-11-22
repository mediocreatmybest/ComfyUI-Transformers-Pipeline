import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

let extension;
function element(tag) {
    return { tag, style: {}, children: [], value: '', textContent: '',
        append(...items) { this.children.push(...items); },
        replaceChildren(...items) { this.children = items; },
        setAttribute(name, value) { this[name] = value; } };
}
const nodes = new Map();
const events = new EventTarget();
const app = { registerExtension(value) { extension = value; },
    graph: { getNodeById(id) { return nodes.get(String(id)); } }, nodeOutputs: {} };
const source = fs.readFileSync(new URL('../web/feedback.js', import.meta.url), 'utf8');
vm.runInNewContext(source.replace(/^import .*;$/gm, ''), {
    app, api: { apiURL: path => path, addEventListener: events.addEventListener.bind(events) }, document: { createElement: element }, URLSearchParams,
});
let forwarded, redraws = 0;
const node = { onExecuted(...args) { forwarded = [this, ...args]; return 17; },
    setDirtyCanvas() { redraws++; }, id: 1, comfyClass: 'CaptionPreviewTpl', widgets: [],
    addDOMWidget(name, type, element, options) {
        const widget = { name, type, element, options };
        this.widgets.push(widget);
        return widget;
    } };
nodes.set('1', node);
extension.nodeCreated(node);
assert.equal(node.widgets.length, 1);
const widget = node.widgets[0];
assert.equal(widget.options.serialize, false);
const [text, gallery] = widget.element.children;
assert.equal(text.readOnly, true);
extension.onNodeOutputsUpdated({ '1': { text: ['<script>not HTML</script>', 'profile facts'],
    tpl_images: [{ filename: 'thumb.png', subfolder: 'transformers_preview', type: 'temp', label: '1. café.png' }] } });
assert.equal(text.value, '<script>not HTML</script>\nprofile facts');
assert.equal(gallery.children.length, 1);
assert.equal(gallery.children[0].children[1].textContent, '1. café.png');
assert.match(gallery.children[0].children[0].src, /^\/view\?/);
extension.onNodeOutputsUpdated({ '1': { text: ['next page'], tpl_images: [] } });
assert.equal(gallery.children.length, 0);
assert.equal(text.value, 'next page');
extension.nodeCreated({ comfyClass: 'UnrelatedNode', addDOMWidget() { throw Error('unrelated node changed'); } });
app.nodeOutputs = { '1': { text: ['cached output'], tpl_images: [] } };
extension.afterConfigureGraph();
assert.equal(text.value, 'cached output');
console.log('Preview UI hooks, safe text, image labels, paging and non-serialization passed.');
const filter = { ...node, id: 2, comfyClass: 'CaptionFilterTpl',
    widgets: ['group_id', 'label', 'minimum_score', 'no_score'].map(name => ({name})) };
extension.nodeCreated(filter);
assert.equal(filter.widgets[0].label, 'Which group of labels?');
assert.equal(filter.widgets[2].label, 'Minimum Score');
assert.equal(filter.widgets[3].label, 'No Score Available');
assert.equal(filter.widgets[4].options.serialize, false);

const unsafeCaption = '<img src=x onerror=alert(1)>\n雪';
extension.onNodeOutputsUpdated({ '1': {
    text: ['Plain text fallback with internal ID'],
    tpl_summary: ['Preview of 2 images'],
    tpl_cards: [
        { filename: 'first.png', image: { filename: 'one.png', subfolder: 'temp & folder', label: 'first.png' },
            captions: [{ label: 'Factual · Original', text: unsafeCaption, tags: ['night'] }],
            classifications: ['Medium: photo'], camera: 'Sony · 55 mm', warnings: [],
            details: 'ID: internal' },
        { filename: 'second.png', captions: [], classifications: [], warnings: ['Image unavailable'] },
    ],
} });
assert.equal(text.hidden, true);
assert.equal(gallery.children.length, 3);
const [, first, second] = gallery.children;
const [image, captionList] = first.children[1].children;
assert.equal(first.children[0].textContent, 'first.png');
assert.match(image.src, /subfolder=temp\+%26\+folder/);
assert.equal(captionList.children[0].children[1].textContent, unsafeCaption);
assert.equal(first.children.at(-1).tag, 'details');
assert.equal(first.children.at(-1).children[0].textContent, 'Show Details');
assert.equal(second.children[0].textContent, 'second.png');
assert.equal(second.children.at(-1).textContent, 'Image unavailable');
extension.onNodeOutputsUpdated({ '1': { text: ['Fallback again'] } });
assert.equal(text.hidden, false);
assert.equal(text.value, 'Fallback again');

const loader = { ...node, id: 3, comfyClass: 'ModelLoaderTpl',
    widgets: ['model_preset', 'model_name_or_path', 'task'].map(name => ({ name })) };
extension.nodeCreated(loader);
assert.deepEqual(loader.widgets.slice(0, 3).map(w => [w.name, w.label]), [
    ['model_preset', 'Model Preset'], ['model_name_or_path', 'Custom Model ID or Path'], ['task', 'Custom Task'],
]);
const records = { ...node, id: 4, comfyClass: 'CaptionRecordsTpl',
    widgets: ['caption_style', 'profile_ids', 'prompt_override'].map(name => ({ name })) };
extension.nodeCreated(records);
assert.deepEqual(records.widgets.slice(0, 3).map(w => w.label), ['Caption Style', 'Custom Profile IDs', 'Prompt Override']);

// Live execution mutates one output entry and calls the node, bypassing the bulk hook.
const live = { text: ['live caption'], tpl_cards: [{ filename: 'live.png' }] };
app.nodeOutputs['1'] = live;
assert.equal(node.onExecuted(live, 'extra'), 17);
assert.deepEqual(forwarded, [node, live, 'extra']);
assert.equal(text.value, 'live caption');
assert.equal(gallery.children.at(-1).children[0].textContent, 'live.png');
assert.ok(redraws > 0);

// Vue frontends with a registered customtext renderer use an empty native textarea
// for that type. A bespoke DOM type mounts the supplied gallery in both renderers.
assert.notEqual(widget.type, 'customtext');
assert.equal(widget.options.hideOnZoom, false);
assert.equal(widget.element['data-capture-wheel'], 'true');

// A cache notification only contains IDs. Read the active graph's existing output.
extension.setup();
const cached = { text: ['cached caption'], tpl_cards: [{ filename: 'cached.png' }] };
app.nodeOutputs = { '1': cached };
events.dispatchEvent(Object.assign(new Event('execution_cached'), { detail: { nodes: ['1'] } }));
assert.equal(text.value, 'cached caption');
const cachedCard = gallery.children.at(-1);
extension.onNodeOutputsUpdated(structuredClone(app.nodeOutputs));
assert.equal(gallery.children.at(-1), cachedCard, 'replay must retain the mounted card and disclosure state');

// Workflow activation can finish after graph configuration.
app.nodeOutputs = { '1': { text: ['restored from history'] } };
extension.afterLoadGraph();
assert.equal(text.value, 'restored from history');

extension.onNodeOutputsUpdated({ '1': { text: ['No captions on this page'], tpl_cards: [] } });
assert.equal(text.hidden, false);
assert.equal(text.value, 'No captions on this page');
