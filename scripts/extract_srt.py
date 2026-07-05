# Extract subtitle text lines from an SRT file and dump as JSON list.
import json, re, sys

srt_path = r"D:\voxtrans\target\debug\output\【今夜解禁】選ばれしホスト業界のレジェンド集結──『ホストコール』ついに始動【HOSTCALL(ホストコール)#001】_1783170641424-p5vvmw\src.srt"
out_path = r"D:\nagisa-rs\models\srt_lines.json"

with open(srt_path, encoding="utf-8-sig") as f:
    content = f.read()

blocks = re.split(r"\r?\n\r?\n", content.strip())
lines = []
for blk in blocks:
    parts = re.split(r"\r?\n", blk)
    # parts[0] = index, parts[1] = times, parts[2:] = text
    if len(parts) >= 3:
        text = "".join(parts[2:]).strip()
        if text:
            lines.append(text)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(lines, f, ensure_ascii=False)

print(f"extracted {len(lines)} subtitle lines -> {out_path}")
# show first few
for i, l in enumerate(lines[:5]):
    print(f"  [{i}] {l}")
print("  ...")
for i, l in enumerate(lines[-3:], start=len(lines)-3):
    print(f"  [{i}] {l}")
