"""Heuristic checks for accidental local provenance in public workflow JSON."""
import ipaddress
import json
import re


PATTERNS = {
    'absolute local path': r'(?i)(?:\b[a-z]:[/\\]|\\\\[\w.-]+\\|(?<![\w.:/])/(?:home|users|root|mnt|volumes|tmp|opt|var|srv|etc)/)',
    'email address': r'(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b',
    'local hostname': r'(?i)(?:\b(?:[\w-]+\.(?:local|lan|internal)|(?:desktop|laptop)-[\w-]+|localhost)\b|https?://[\w-]+(?=[:/\s]|$))',
    'personal or machine identifier': r'(?i)\b(?:user(?:name)?|host(?:name)?|computer(?:name)?|machine(?:name)?|created by|author)\s*[:=]\s*\S+',
    'hardware identity': r'(?i)\b(?:nvidia|geforce|quadro|radeon|(?:rtx|gtx|tesla)\s*[- ]?\d+[\w -]*|(?:a|h)100|amd\s+(?:gpu|ryzen)|intel\s+(?:gpu|arc|core))\b',
    'development agent': r'(?i)\b(?:codex|claude code|agent[_ -](?:id|session|instructions)|agents\.md|claude\.md|gemini\.md)\b',
    'runtime provenance': r'(?i)\b(?:desktop\s+(?:transformers\s+)?\d+\.\d+|(?:transformers|python|pytorch|torch|accelerate)\s*[:= ]\s*v?\d+\.\d+\.\d+|isolated\s+(?:\w+\s+){0,3}(?:environment|runtime|worker))\b',
    'validation provenance': r'(?i)\b(?:physically validated|(?:tested|validated)\s+on\b|(?:local|development)\s+(?:workstation|machine|environment))',
}
PROVENANCE_KEYS = {
    'username', 'user_name', 'hostname', 'host_name', 'computer_name', 'machine_name',
    'email', 'created_by', 'author', 'supplied_author', 'owner', 'agent', 'agent_id', 'session_id',
    'node_versions', 'runtime', 'environment', 'provenance', 'validation_machine',
}


def privacy_issues(value, path='$'):
    """Inspect keys, values and JSON embedded in widgets; return locations/reasons.

    This catches common leaks, not arbitrary names in natural-language prose.
    Repository IDs and useful generic compatibility guidance remain permitted.
    """
    issues = []
    if isinstance(value, dict):
        for key, child in value.items():
            location = f'{path}.{key}'
            if key.lower() in PROVENANCE_KEYS and child not in (None, '', {}, []):
                issues.append((location, 'personal/development metadata'))
            issues.extend(privacy_issues(key, location + ' (key)'))
            issues.extend(privacy_issues(child, location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            issues.extend(privacy_issues(child, f'{path}[{index}]'))
    elif isinstance(value, str):
        # Decode nested JSON so escaped Windows paths and metadata are checked too.
        if value.lstrip().startswith(('{', '[')):
            try:
                embedded = json.loads(value)
            except ValueError:
                pass
            else:
                return privacy_issues(embedded, path + ' (embedded JSON)')
        repository_id = re.fullmatch(r'[\w.-]+/[\w.-]+', value)
        for reason, pattern in PATTERNS.items():
            if repository_id and reason in {'hardware identity', 'development agent'}:
                continue
            if re.search(pattern, value):
                issues.append((path, reason))
        for token in re.findall(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])|[\da-fA-F]*:[\da-fA-F:]+', value):
            try:
                address = ipaddress.ip_address(token)
            except ValueError:
                continue
            if address.is_private or address.is_loopback or address.is_link_local:
                issues.append((path, 'private network address'))
    return issues
