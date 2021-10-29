def flat_dict(data: dict) -> dict:
    out = dict()
    for key, value in out.items():
        if isinstance(value, dict):
            out.update(flat_dict)
        else:
            out[key] = value
    return out
