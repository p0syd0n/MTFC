def clean(filename, blacklist=None):
    if blacklist is None:
        blacklist = []
    
    # Normalize blacklist to lowercase for case-insensitive matching
    blacklist = [item.lower().strip() for item in blacklist]

    with open(filename, "r") as file:
        lines = [line.strip() for line in file if line.strip()]

    if not lines:
        return

    all_columns = [col.strip() for col in lines[0].split(",")]
    data_rows = lines[1:]

    # 1. Identify indices that are NOT in the blacklist
    valid_indices = [
        i for i, col in enumerate(all_columns) 
        if col.lower() not in blacklist
    ]

    new_lines = []
    numeric_indices = None

    for line in data_rows:
        fields = [f.strip() for f in line.split(",")]
        
        # Ensure we don't crash on rows with fewer columns than the header
        if len(fields) < len(all_columns):
            continue

        # 2. On the first valid row, further filter for numeric-only columns
        if numeric_indices is None:
            numeric_indices = []
            for i in valid_indices:
                try:
                    float(fields[i])
                    numeric_indices.append(i)
                except ValueError:
                    continue

        # 3. Skip lines with missing data in the target numeric columns
        if any(fields[i] == "" for i in numeric_indices):
            print(f"Skipping incomplete line: {line}")
            continue

        # Extract only the allowed numeric fields
        new_lines.append(",".join(fields[i] for i in numeric_indices))

    # Update columns to match the final selection
    final_columns = [all_columns[i] for i in numeric_indices]

    # 4. Write back to file
    with open(filename, "w") as file:
        file.write(",".join(final_columns) + "\n")
        file.write("\n".join(new_lines) + "\n")

# Usage
blacklist_cols = ["close_forcast"]
clean("g_train.csv", blacklist=blacklist_cols)