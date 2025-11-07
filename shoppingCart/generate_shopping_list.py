import json

data = []
with open("recipes_samples.json", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():  # skip empty lines
            data.append(json.loads(line))

# Example: print title of first recipe
print(data[0]["title"])