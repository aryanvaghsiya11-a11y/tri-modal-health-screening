import json

with open("tri-modal-by-aryan.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("tri-modal-stripped.py", "w", encoding="utf-8") as f:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            f.write("".join(cell.get("source", [])))
            f.write("\n\n")
