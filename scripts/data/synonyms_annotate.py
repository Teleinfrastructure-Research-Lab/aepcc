import tkinter as tk
import pandas as pd

# List of synonym options (buttons)
SYNONYMS = [
    "chair",
    "electronics",
    "bed",
    "bench",
    "curtain",
    "bookshelf",
    "containers",
    "cabinet",
    "clock",
    "sofa",
    "counter",
    "table",
    "lamp",
    "kitchen",
    "monitor",
    "wallattached",
    "pillow",
    "sink"
]
CSV_PATH = "outputs\cats_csvs\mt_cats.csv"

class SynonymLabeler:
    def __init__(self, root, csv_path):
        self.root = root
        self.root.title("Synonym Labeler")

        # Load CSV into DataFrame
        self.df = pd.read_csv(csv_path)
        
        # Ensure 'synonym' column exists. If it doesn’t, create an empty column.
        if "synonym" not in self.df.columns:
            self.df["synonym"] = ""

        self.csv_path = csv_path
        self.current_index = 0
        self.total_rows = len(self.df)

        # Create GUI elements
        self.create_widgets()
        self.update_display()

    def create_widgets(self):
        # Frame for top info
        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(pady=10)

        # Label to show row info and category
        self.row_info_label = tk.Label(self.info_frame, text="", font=("Arial", 12))
        self.row_info_label.pack()

        # Frame for synonym buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=10)

        # Dynamically create a button for each synonym
        for synonym in SYNONYMS:
            btn = tk.Button(
                self.button_frame,
                text=synonym,
                command=lambda syn=synonym: self.set_synonym(syn)
            )
            btn.pack(side=tk.LEFT, padx=5)

        # Frame for navigation & save/quit buttons
        self.nav_frame = tk.Frame(self.root)
        self.nav_frame.pack(pady=10)

        self.back_button = tk.Button(self.nav_frame, text="Back", command=self.previous_entry)
        self.back_button.pack(side=tk.LEFT, padx=10)

        self.skip_button = tk.Button(self.nav_frame, text="Skip", command=self.skip_entry)
        self.skip_button.pack(side=tk.LEFT, padx=10)

        self.next_button = tk.Button(self.nav_frame, text="Next", command=self.next_entry)
        self.next_button.pack(side=tk.LEFT, padx=10)

        # Save & Quit button
        self.save_quit_button = tk.Button(self.nav_frame, text="Save & Quit", command=self.save_and_quit)
        self.save_quit_button.pack(side=tk.LEFT, padx=10)

    def update_display(self):
        # Current row number out of total
        row_number = self.current_index + 1

        # Get category from DataFrame
        category_value = self.df.at[self.current_index, "category"]

        # Get current synonym
        current_synonym = self.df.at[self.current_index, "synonym"]

        # Build display text
        display_text = f"Row {row_number}/{self.total_rows} | Category: {category_value}"
        if current_synonym:
            display_text += f" | Synonym: {current_synonym}"

        self.row_info_label.config(text=display_text)

    def set_synonym(self, synonym):
        # Set the synonym in the DataFrame for the current row
        self.df.at[self.current_index, "synonym"] = synonym
        self.update_display()

    def skip_entry(self):
        # Assign empty string and then move to next row
        self.df.at[self.current_index, "synonym"] = ""
        self.next_entry()

    def next_entry(self):
        # Move to the next row if not at the end
        if self.current_index < self.total_rows - 1:
            self.current_index += 1
            self.update_display()

    def previous_entry(self):
        # Move to the previous row if not at the beginning
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()

    def save_and_quit(self):
        """
        Save the current DataFrame to a new CSV file and quit the application immediately.
        """
        save_path = CSV_PATH  # Or choose a path you prefer
        self.df.to_csv(save_path, index=False)
        self.root.destroy()  # Close the GUI window

def main():
    root = tk.Tk()
    app = SynonymLabeler(root, csv_path=CSV_PATH)
    root.mainloop()

if __name__ == "__main__":
    main()