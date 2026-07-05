# Run Python nagisa.wakati over all extracted subtitle lines and dump results.
import json
import nagisa

with open(r"D:/nagisa-rs/models/srt_lines.json", encoding="utf-8") as f:
    lines = json.load(f)

out = {}
for i, line in enumerate(lines):
    out[i] = nagisa.wakati(line)

with open(r"D:/nagisa-rs/models/srt_wakati_py.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)

# quick sanity: count total words
total = sum(len(v) for v in out.values())
print(f"processed {len(lines)} lines, {total} total words")
