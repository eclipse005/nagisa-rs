# Run Python nagisa.tagging over all extracted subtitle lines and dump results.
import json
import nagisa

with open(r"D:/nagisa-rs/models/srt_lines.json", encoding="utf-8") as f:
    lines = json.load(f)

out = {}
for i, line in enumerate(lines):
    r = nagisa.tagging(line)
    out[i] = {"words": r.words, "postags": r.postags}

with open(r"D:/nagisa-rs/models/srt_tagging_py.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)

total = sum(len(v["words"]) for v in out.values())
print(f"processed {len(lines)} lines, {total} total words (with postags)")
