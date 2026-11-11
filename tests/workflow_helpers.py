"""Validate editor graphs and derive their executable API prompts for regression tests."""
from copy import deepcopy


def editor_to_api(workflow, definitions):
    if workflow.get('version') != 0.4:
        raise ValueError('Expected ComfyUI editor workflow version 0.4')
    nodes = {node['id']: node for node in workflow['nodes']}
    if len(nodes) != len(workflow['nodes']):
        raise ValueError('Duplicate node id')
    links = {link[0]: link for link in workflow['links']}
    if len(links) != len(workflow['links']):
        raise ValueError('Duplicate link id')
    prompt, incoming, outgoing, dependencies = {}, {}, {}, {}
    for node_id, node in nodes.items():
        definition = definitions[node['type']]
        if node.get('mode', 0) != 0:
            raise ValueError('Examples must contain active nodes')
        schema = definition['input']
        fields = {**schema.get('required', {}), **schema.get('optional', {})}
        widget_fields = [(name, spec) for name, spec in fields.items()
                         if isinstance(spec[0], list) or spec[0] in {'STRING', 'INT', 'FLOAT', 'BOOLEAN'}]
        values = node.get('widgets_values', [])
        if len(values) > len(widget_fields):
            raise ValueError('Unexpected widget values')
        inputs = {}
        for index, (name, spec) in enumerate(widget_fields):
            options = spec[1] if len(spec) > 1 else {}
            default = options.get('default', spec[0][0] if isinstance(spec[0], list) else None)
            value = values[index] if index < len(values) else default
            if value is not None:
                valid = (value in spec[0] if isinstance(spec[0], list) else
                         isinstance(value, str) if spec[0] == 'STRING' else
                         type(value) is bool if spec[0] == 'BOOLEAN' else
                         type(value) is int if spec[0] == 'INT' else type(value) in {int, float})
                if not valid or ('min' in options and value < options['min']) or ('max' in options and value > options['max']):
                    raise ValueError(f'Invalid widget {node_id}.{name}: {value!r}')
                inputs[name] = deepcopy(value)
        for slot, port in enumerate(node.get('inputs', [])):
            if port['name'] not in fields:
                raise ValueError('Unknown input port')
            if port.get('link') is not None:
                incoming[node_id, slot] = port['link']
        for slot, port in enumerate(node.get('outputs', [])):
            if slot >= len(definition['output']) or port['type'] != definition['output'][slot]:
                raise ValueError('Invalid output port')
            outgoing[node_id, slot] = set(port.get('links') or [])
        prompt[str(node_id)] = {'class_type': node['type'], 'inputs': inputs,
                                '_meta': {'title': node.get('title', node['type'])}}
        dependencies[node_id] = set()
    used_inputs, used_outputs = set(), {}
    for link_id, origin, output_slot, target, input_slot, kind in links.values():
        if origin not in nodes or target not in nodes:
            raise ValueError('Link references an unknown node')
        source_port = nodes[origin]['outputs'][output_slot]
        target_port = nodes[target]['inputs'][input_slot]
        schema = definitions[nodes[target]['type']]['input']
        spec = {**schema.get('required', {}), **schema.get('optional', {})}[target_port['name']]
        target_kind = 'STRING' if isinstance(spec[0], list) else spec[0]
        if source_port['type'] != kind or target_port['type'] != kind or target_kind != kind:
            raise ValueError('Incompatible link types')
        if incoming.get((target, input_slot)) != link_id or link_id not in outgoing.get((origin, output_slot), set()):
            raise ValueError('Link is not mirrored in its ports')
        if (target, input_slot) in used_inputs:
            raise ValueError('Multiple links target one input')
        used_inputs.add((target, input_slot))
        used_outputs.setdefault((origin, output_slot), set()).add(link_id)
        prompt[str(target)]['inputs'][target_port['name']] = [str(origin), output_slot]
        dependencies[target].add(origin)
    if used_inputs != set(incoming) or any(used_outputs.get(key, set()) != value for key, value in outgoing.items()):
        raise ValueError('Dangling port link')
    for node_id, node in nodes.items():
        required = definitions[node['type']]['input'].get('required', {})
        if set(required) - prompt[str(node_id)]['inputs'].keys():
            raise ValueError('Missing required inputs')
    pending = deepcopy(dependencies)
    while pending:
        ready = {node_id for node_id, deps in pending.items() if not deps}
        if not ready:
            raise ValueError('Workflow contains a cycle')
        pending = {node_id: deps - ready for node_id, deps in pending.items() if node_id not in ready}
    if not any(definitions[node['type']].get('output_node') for node in nodes.values()):
        raise ValueError('Workflow has no output node')
    return prompt
