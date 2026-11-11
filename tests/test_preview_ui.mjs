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
const app = { registerExtension(value) { extension = value; },
    graph: { getNodeById(id) { return nodes.get(String(id)); } }, nodeOutputs: {} };
const source = fs.readFileSync(new URL('../web/feedback.js', import.meta.url), 'utf8');
vm.runInNewContext(source.replace(/^import .*;$/gm, ''), {
    app, api: { apiURL: path => path }, document: { createElement: element }, URLSearchParams,
});
const node = { id: 1, comfyClass: 'CaptionPreviewTpl', widgets: [],
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
