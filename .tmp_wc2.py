from pathlib import Path
import re
p=Path(r'd:/tox-agent/docs/papers/ToxAgent_IEEE_ACM_Full_Paper_vi.md')
text=p.read_text(encoding='utf-8')
words=re.findall(r"\b\w+\b", text, flags=re.UNICODE)
print('WORD_COUNT', len(words))
