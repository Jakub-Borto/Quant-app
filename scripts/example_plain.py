"""Example plain Python quick script.

No `# app: streamlit` marker, so the Scripts module runs it as
`python -u example_plain.py` — no browser; prints stream live into the
module's console and stay there (until dismissed) after the script ends.
"""

import time

print("Hello from a plain quick script.")
for i in range(5):
    print(f"working... {i + 1}/5")
    time.sleep(0.5)
print("done.")
