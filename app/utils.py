import bleach

ALLOWED_TAGS = list(bleach.sanitizer.ALLOWED_TAGS) + [
    'p', 'pre', 'span', 'hr', 'br', 'div', 'mark'
]
ALLOWED_ATTRIBUTES = {
    '*': ['class', 'style'],
    'a': ['href', 'rel', 'target'],
}


def sanitize_html(text: str) -> str:
    if not text:
        return ''
    return bleach.clean(text, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRIBUTES)
