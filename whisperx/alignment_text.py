"""Generic spoken/display text mapping. Callers decide how numbers are spoken."""
import re


def prepare_alignment_text(text, dictionary, no_spaces=False, replacements=()):
    """Expand explicit spans before CTC while retaining display character indices.

    A replacement must fit in one display word. Its full spoken interval maps
    to the first display character; other display characters have no independent
    phonetic boundary. Offsets are Python Unicode character offsets, end exclusive.
    """
    cursor = 0
    expanded = []
    origins = []
    mapped_spans = []
    for replacement in sorted(replacements, key=lambda r: r["start"]):
        start, end, spoken = replacement["start"], replacement["end"], replacement["text"]
        if (not isinstance(start, int) or not isinstance(end, int) or
                not cursor <= start < end <= len(text) or
                not isinstance(spoken, str) or not spoken.strip() or
                any(c.isspace() for c in text[start:end])):
            raise ValueError("Spoken replacements must be nonempty, nonoverlapping spans within a display word")
        expanded.extend(text[cursor:start])
        origins.extend(range(cursor, start))
        expanded.extend(spoken)
        origins.extend([start] * len(spoken))
        mapped_spans.append((start, end))
        cursor = end
    expanded.extend(text[cursor:])
    origins.extend(range(cursor, len(text)))
    chars, indices, unsupported = [], [], []
    first = len(expanded) - len("".join(expanded).lstrip())
    last = len("".join(expanded).rstrip())
    for index, char in enumerate(expanded):
        if not first <= index < last:
            continue
        normalized = char.lower().replace(" ", "|") if not no_spaces else char.lower()
        for token in normalized:
            if token in dictionary or token not in {" ", "|"}:
                chars.append(token)
                indices.append(origins[index])
                if token not in dictionary:
                    unsupported.append({"character": token, "index": origins[index]})
    words = []
    unsupported_indices = {char["index"] for char in unsupported}
    for match in re.finditer(r"\S" if no_spaces else r"\S+", text):
        unknown = any(index in unsupported_indices for index in range(match.start(), match.end()))
        spoken = any(start < match.end() and end > match.start() for start, end in mapped_spans)
        words.append({"word": match.group(), "start_index": match.start(), "end_index": match.end(),
                      "method": "wildcard" if unknown else "spoken" if spoken else "ctc"})
    return chars, indices, {"unsupported_characters": unsupported, "words": words}
