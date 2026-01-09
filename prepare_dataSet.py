import os
import json

folder_path = r"C:\Users\Asghar Qambar Rizvi\OneDrive\Desktop\LLM-Book-Helper\data_Set_json"

merged_no_filename_path = os.path.join(folder_path, "merged_dataset_no_filenames.json")
contexts_only_path = os.path.join(folder_path, "contexts_and_sources.json")

skip_names = {os.path.basename(merged_no_filename_path), os.path.basename(contexts_only_path)}

merged_no_filename = {}
contexts_only = {}

current_index = 0
file_counts = {}     
errors = []

all_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".json")])

for filename in all_files:
    if filename in skip_names:
        continue

    full_path = os.path.join(folder_path, filename)
    print(f"Reading {filename} ...")

    try:
        with open(full_path, "r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except Exception as e:
        errors.append((filename, str(e)))
        print(f"  ! Failed to load {filename}: {e}")
        continue

    count_this_file = 0

    if isinstance(data, dict):
        try:
            items = sorted(data.items(), key=lambda kv: int(kv[0]))
        except Exception:
            items = list(data.items())

        for key, value in items:
            merged_no_filename[str(current_index)] = {
                "generated": value.get("generated", []),
                "context": value.get("context", "")
            }
            contexts_only[str(current_index)] = {
                "context": value.get("context", ""),
                "law_data_set_name": filename
            }
            current_index += 1
            count_this_file += 1

    elif isinstance(data, list):
        for value in data:
            merged_no_filename[str(current_index)] = {
                "generated": value.get("generated", []),
                "context": value.get("context", "")
            }
            contexts_only[str(current_index)] = {
                "context": value.get("context", ""),
                "law_data_set_name": filename
            }
            current_index += 1
            count_this_file += 1
    else:
        print(f"  ! Unexpected top-level JSON type in {filename}: {type(data)} (skipping)")
        continue

    file_counts[filename] = count_this_file
    print(f"  → {count_this_file} entries added from {filename}")

print("\nWriting output files ...")
with open(merged_no_filename_path, "w", encoding="utf-8") as f:
    json.dump(merged_no_filename, f, indent=2, ensure_ascii=False)

with open(contexts_only_path, "w", encoding="utf-8") as f:
    json.dump(contexts_only, f, indent=2, ensure_ascii=False)

print("\nDone.")
print(f"Total merged entries: {current_index}")
print("Per-file counts:")
for fn, cnt in file_counts.items():
    print(f"  - {fn}: {cnt}")

if errors:
    print("\nFiles skipped due to errors:")
    for fn, err in errors:
        print(f"  - {fn}: {err}")

print(f"\nOutputs created:\n  - {merged_no_filename_path}\n  - {contexts_only_path}")
