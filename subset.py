import json

# Paths
json_path = 'archive/similar/test_similar.json'
subset_output_path = 'archive/similar/test_subset_styletransfer.json'

# Load the full JSON
with open(json_path, 'r') as f:
    data = json.load(f)

# Extract only the "style_transfer" entries
style_data = data.get("style_transfer", {})

# Limit to first 5000 items
subset_keys = list(style_data.keys())[:5000]
subset_data = {k: style_data[k] for k in subset_keys}

# Wrap it back in the same outer format
subset_json = {"style_transfer": subset_data}

# Save subset
with open(subset_output_path, 'w') as f:
    json.dump(subset_json, f, indent=4)

print(f"Saved {len(subset_data)} pairs to subset JSON.")
