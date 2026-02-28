def clean(filename):
    with open(filename, "r") as file:
        data = file.read()
        lines = data.split("\n")
        lines = [line for line in lines if len(line) != 0]

    columns = lines[0].split(",")
    del lines[0]

    new_lines = []
    numeric_indices = None

    for line in lines:
        fields = line.split(",")

        # On the first row, determine which columns are numeric
        if numeric_indices is None:
            numeric_indices = []
            for i, field in enumerate(fields):
                try:
                    float(field.strip())
                    numeric_indices.append(i)
                except ValueError:
                    pass

        # Skip lines with missing data
        if any(fields[i].strip() == "" for i in numeric_indices):
            print(f"Skipping incomplete line: {line}")
            continue

        new_lines.append(",".join(fields[i] for i in numeric_indices))

    columns = [columns[i] for i in numeric_indices]

    with open(filename, "w") as file:
        end_content = ",".join(columns) + "\n"
        for new_line in new_lines:
            end_content += new_line + "\n"
        file.write(end_content)

clean("jpm_test.csv")