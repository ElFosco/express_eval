import ast
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.table import Table


def plot_sudoku_explanations(all_rows, col_title=None, additional_titles=None,
                             col_grid="grid", col_hint_facts="hint_facts", col_hint_constraints="hint_constraints",
                             col_hint_derived="hint_derived"):
    if isinstance(all_rows, pd.DataFrame):
        all_rows = [row for _, row in all_rows.iterrows()]

    # Number of subplots: One row, multiple columns
    n = len(all_rows)
    fig, axes = plt.subplots(1, n, figsize=(n * 6, 3))  # Adjust the figure size depending on the number of plots

    # In case there is only one plot, axes will not be iterable, so we make it iterable
    if n == 1:
        axes = [axes]

    # Loop through all rows and corresponding axes for plotting
    for i, (row_df, ax) in enumerate(zip(all_rows, axes)):
        ax.axis('off')  # Turn off the axis

        title = ""
        if col_title is not None and isinstance(col_title, list):
            title = "\n".join([sub_col_title + "=" + str(row_df[sub_col_title]) for sub_col_title in col_title])
        elif col_title is not None:
            title = row_df[col_title]
        if additional_titles is not None:
            for additional_title in additional_titles:
                title += f" - {additional_title}"
        ax.set_title(title, fontsize=9)

        # Parse the sudoku grid
        if isinstance(row_df[col_grid], np.ndarray):
            sudoku_grid = row_df[col_grid]
        else:
            data_str = row_df[col_grid].replace("[", "").replace("]", "").strip()
            rows = data_str.split("\n")
            data = [list(map(int, row.split())) for row in rows]
            sudoku_grid = np.array(data)

        # Create the table for displaying the grid
        table = Table(ax, bbox=(0, 0, 0.8, 1))
        ax.add_table(table)

        if isinstance(row_df[col_hint_facts], str):
            hint_facts = ast.literal_eval(row_df[col_hint_facts])
            hint_constraints = ast.literal_eval(row_df[col_hint_constraints])
            hint_derived = ast.literal_eval(row_df[col_hint_derived])
        else:
            # Extract the facts, constraints, and derived values
            hint_facts = list_to_tuple(row_df[col_hint_facts])
            hint_constraints = list_to_tuple(row_df[col_hint_constraints])
            hint_derived = to_tuple(row_df[col_hint_derived])

        all_facts = {(row, col): val for fact_type, (row, col, val) in hint_facts}
        all_row_cons = [hint for hint_type, hint in hint_constraints if hint_type == "ROW"]
        all_col_cons = [hint for hint_type, hint in hint_constraints if hint_type == "COL"]
        all_block_cons = [(i, j) for hint_type, hint in hint_constraints if hint_type == "BLOCK" for i in
                          range(hint[0][0], hint[1][0]) for j in range(hint[0][1], hint[1][1])]

        _, (derived_row, derived_col, derived_val) = hint_derived

        nrows = sudoku_grid.shape[0]
        nblock_offset = int(nrows ** (1 / 2))

        # Fill cells in the table
        for row in range(nrows):
            for col in range(nrows):
                face_color = "w"

                # Add cell to the table
                cell = table.add_cell(row, col, width=1, height=1, facecolor=face_color, fill=True,
                                      text=sudoku_grid[row, col], loc="center")
                cell.set(fontsize=12)
                # Facts and derived values
                if (row, col) in all_facts:
                    cell.get_text().set_color('green')
                elif (row, col) == (derived_row, derived_col):
                    cell.get_text().set_color('red')
                    cell.get_text().set_fontweight('bold')
                    cell.get_text().set_text(derived_val)
                else:
                    cell.get_text().set_color('lightgrey')
                    cell.get_text().set_text(sudoku_grid[row, col])

                # Row, column, and block constraints
                visible_edges = set()

                if row in all_row_cons and col in all_col_cons:
                    visible_edges |= {"R", "B", "T", "L"}
                elif row in all_row_cons:
                    visible_edges |= {"B", "T"}
                    if col == 0:
                        visible_edges |= {"L"}
                    elif col + 1 == nrows:
                        visible_edges |= {"R"}
                elif col in all_col_cons:
                    visible_edges |= {"R", "L"}
                    if row == 0:
                        visible_edges |= {"T"}
                    elif row + 1 == nrows:
                        visible_edges |= {"B"}

                if (row, col) in all_block_cons:
                    if row % nblock_offset == 0:
                        visible_edges |= {"T"}
                    if col % nblock_offset == 0:
                        visible_edges |= {"L"}
                    if (row + 1) % nblock_offset == 0:
                        visible_edges |= {"B"}
                    if (col + 1) % nblock_offset == 0:
                        visible_edges |= {"R"}

                cell.visible_edges = "".join(visible_edges)

    plt.tight_layout()  # Adjust layout so subplots do not overlap
    plt.show()

def list_to_tuple(lst):

    return [to_tuple(elem) for elem in lst]

def to_tuple(lst):

    return tuple(to_tuple(x) if isinstance(x, list) else x for x in lst)




import os
import pandas as pd

import os
import re
import pandas as pd


def escape_latex(text: str) -> str:
    return text.replace("_", " ").replace("&", "\\&").replace("%", "\\%").replace("$", "\\$")


def simplify_label(label: str, type_prefix: str) -> str:
    # Mapping normalization names
    mapping = {
        f"{type_prefix}_0": "no normalization",
        f"{type_prefix}_1": "nadir",
        f"{type_prefix}_2": "local",
        f"{type_prefix}_3": "cumulative"
    }

    # Replace based on prefix match
    for raw, pretty in mapping.items():
        if label.startswith(raw):
            label = label.replace(raw, pretty)
            break

    # Remove lr_ values
    label = re.sub(r"lr_[0-9.]+", "", label)

    # Remove "diversification"
    label = label.replace("diversification", "")

    # Replace remaining underscores with space
    label = label.replace("_", " ").strip()

    return escape_latex(label)


def escape_latex(text: str) -> str:
    return text.replace("_", " ").replace("&", "\\&").replace("%", "\\%").replace("$", "\\$")

def simplify_label(label: str, type_prefix: str) -> str:
    mapping = {
        f"{type_prefix}_0": "no normalization",
        f"{type_prefix}_1": "nadir",
        f"{type_prefix}_2": "local",
        f"{type_prefix}_3": "cumulative"
    }

    for raw, pretty in mapping.items():
        if label.startswith(raw):
            label = label.replace(raw, pretty)
            break

    label = re.sub(r"lr_[0-9.]+", "", label)
    label = label.replace("diversification", "")
    label = label.replace("cpucb", "machop")
    label = label.replace("disjunction", "disj.")
    label = label.replace("_", " ").strip()
    label = re.sub(r"\s+", " ", label)  # Replace multiple spaces with one

    return escape_latex(label)





def escape_latex(text: str) -> str:
    return text.replace("_", " ").replace("&", "\\&").replace("%", "\\%").replace("$", "\\$")

def simplify_label(label: str, type_prefix: str) -> str:
    mapping = {
        f"{type_prefix}_norm_0": "no normalization",
        f"{type_prefix}_norm_1": "nadir",
        f"{type_prefix}_norm_2": "local",
        f"{type_prefix}_norm_3": "cumulative"
    }

    for raw, pretty in mapping.items():
        if label.startswith(raw):
            label = label.replace(raw, pretty)
            break

    label = re.sub(r"lr_[0-9.]+", "", label)
    label = label.replace("diversification", "")
    label = label.replace("cpucb", "machop")
    label = label.replace("disjunction", "disj.")
    label = label.replace("_", " ").strip()
    label = re.sub(r"\s+", " ", label)  # Normalize spacing

    return escape_latex(label)

def generate_latex_table(root_folder: str, type_prefix: str) -> str:
    rows = []

    for type_folder in os.listdir(root_folder):
        if not type_folder.startswith(type_prefix):
            continue

        type_path = os.path.join(root_folder, type_folder)
        if not os.path.isdir(type_path):
            continue

        intermediate_path = next((os.path.join(type_path, d) for d in os.listdir(type_path)
                                  if os.path.isdir(os.path.join(type_path, d))), None)
        if intermediate_path is None:
            continue

        for run_folder in os.listdir(intermediate_path):
            run_path = os.path.join(intermediate_path, run_folder)
            if not os.path.isdir(run_path):
                continue

            mean1_list, mean2_list = [], []

            for inner_folder_name in os.listdir(run_path):
                inner_folder = os.path.join(run_path, inner_folder_name)
                dataset_file = os.path.join(inner_folder, "time.csv")

                if os.path.isdir(inner_folder) and os.path.isfile(dataset_file):
                    df = pd.read_csv(dataset_file)
                    time1_mean = df["time exp 1"].dropna().mean()
                    time2_mean = df["time exp 2"].dropna().mean()

                    if pd.notna(time1_mean) and pd.notna(time2_mean):
                        mean1_list.append(time1_mean)
                        mean2_list.append(time2_mean)

            if mean1_list and mean2_list:
                overall_mean1 = pd.Series(mean1_list).mean()
                overall_std1 = pd.Series(mean1_list).std()
                overall_mean2 = pd.Series(mean2_list).mean()
                overall_std2 = pd.Series(mean2_list).std()
                label_raw = f"{type_folder}_{run_folder}"
                label = simplify_label(label_raw, type_prefix)
                rows.append((label, overall_mean1, overall_std1, overall_mean2, overall_std2))

    # Build LaTeX table
    latex_table = "\\begin{tabular}{p{4cm}cccc}\n"
    latex_table += "\\toprule\n"
    latex_table += "Configuration & Mean 1 & Std 1 & Mean 2 & Std 2 \\\\\n"
    latex_table += "\\midrule\n"

    for row in rows:
        latex_table += f"{row[0]} & {row[1]:.2f} & {row[2]:.2f} & {row[3]:.2f} & {row[4]:.2f} \\\\\n"

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}"

    return latex_table
