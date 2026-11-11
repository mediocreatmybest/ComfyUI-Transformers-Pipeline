def feedback(result, text):
    """Keep socket results intact while sending bounded, transient UI feedback."""
    return {"ui": {"text": [str(text)[:24000]]}, "result": result}
