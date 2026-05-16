#!./tensorflow/bin/python

import sys
import re

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <whitelist_file>")
    sys.exit(1)

whitelist_path = sys.argv[1]

# Load whitelist (case-insensitive)
with open(whitelist_path, 'r') as f:
    whitelist = set(line.strip().lower() for line in f if line.strip())

# Regex to capture:
# 1) text inside quotes
# 2) text after the word "identified"
pattern = re.compile(r'"([^"]+)"\s+identified\s+(.+)$')

matches = 0
non_matches = 0
skipped = 0

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    m = pattern.match(line)
    if not m:
        continue  # skip malformed lines

    quoted_text = m.group(1).strip().lower()
    identified_text = m.group(2).strip().lower()

    # Skip if identified text is not in whitelist
    if identified_text not in whitelist:
        skipped += 1
        continue

    if quoted_text == identified_text:
        matches += 1
    else:
        non_matches += 1

total = matches + non_matches

print(f"Matches: {matches}")
print(f"Non-matches: {non_matches}")
print(f"Skipped (not in whitelist): {skipped}")

if total == 0:
    print("Match ratio: N/A (no valid comparisons)")
else:
    ratio = matches / total
    print(f"Match ratio: {ratio:.6f}")
