import os
from joblib import dump
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from auto_eda import AutoEDA
from regressor import Regressor
from classifier import Classifier
from termcolor import colored

# Dictionary for terminal colors
colors = {
    "input" : "yellow",
    "output" : "blue",
    "error" : "red",
    "success" : "green"
}

# Class for selecting CSV files using Tkinter
class CSVSelector:
    # Method to open the file explorer and let the user choose a CSV file
    @staticmethod
    def load_csv():
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

        return file_path

    # Tkinter settings such as title and window size
    def __init__(self, master):
        self.master = master
        master.title("CSV Selector")
        master.geometry("100x100")

        self.selected_file = None
        self.process_csv()

    # The main method handling CSV files using AutoEDA, Regressor, and Classifier classes
    def process_csv(self):
        # Prompt user to choose a CSV file
        self.selected_file = CSVSelector.load_csv()
        if self.selected_file:
            try:
                # Read CSV file into DataFrame
                df = pd.read_csv(self.selected_file)
                print("*"*4, f"DataFrame for {self.selected_file.split('/')[-1]}", "*"*4)
                print(df.head(10))
                target = input(colored("Enter target column : ", colors["input"]))
                # Perform Automated EDA using AutoEDA class
                ada = AutoEDA(df, target=target)
                ada.get_na()

                ada.regressor_classifier()
                # Handle categorical columns if present
                if ada.dummies_:
                    print(colored(f"{len(ada.dummies_list_)}/{len(ada.df.columns)} columns were categorical --> creating dummies on those columns.", colors["error"]))
                    ada.binary_features()
                    print(colored(f"Dummies created successfully", colors["success"]))

                # Handle missing values if present
                if ada.missing_values_:
                    print(colored(f"Missing values found : {ada.missing_values}", colors["error"]))
                    print(colored(f"Total size of data : {len(ada.df)}\n", colors["output"]))
                    if ada.regressor_classifier_:
                        # Ask user how to handle missing data for regression task
                        how_ = int(input(colored("How would you like to handle the missing data?\n"+
                                                "1 : Use Interpolate\n"+
                                                "2 : Use IterativeImputer (regression to predict missing values, only applicable for numeric values)\n"+
                                                "3 : Use SimpleImputer (fill with median, only applicable for numeric values)\n"+
                                                "4 : Use KNNImputer (replaces missing values with mean of k-nearest neighbors values)\n"+
                                                "5 : Drop Them (remove them from the data)\n"+
                                                ">>> ", colors["input"])))
                        
                        # Perform selected method to handle missing data
                        if how_ == 1:
                            ada.fill_na(how="interpolate")
                            os.system("cls || clear")
                            print(colored(f"Missing values after Interpolate() : {ada.df.isna().sum().sum()}", colors["output"]))

                        elif how_ == 2:
                            ada.fill_na(how="iterative_imputer")
                            os.system("cls || clear")
                            print(colored(f"Missing values after IterativeImputer() : {ada.df.isna().sum().sum()}", colors["output"]))

                        elif how_ == 3:
                            ada.fill_na(how="simple_imputer")
                            os.system("cls || clear")
                            print(colored(f"Missing values after SimpleImputer() : {ada.df.isna().sum().sum()}", colors["output"]))
                        
                        elif how_ == 4:
                            ada.fill_na(how="knn_imputer")
                            os.system("cls || clear")
                            print(colored(f"Missing values after KNNImputer() : {ada.df.isna().sum().sum()}", colors["output"]))

                        elif how_ == 5:
                            ada.fill_na(how="drop_na")
                            os.system("cls || clear")
                            print(colored(f"Missing values after drop_na() : {ada.df.isna().sum().sum()}", colors["output"]))

                        else:
                            raise SystemExit("Program exitted...")
                        
                    else:
                        # Ask user how to handle missing data for classification task
                        how_ = int(input(colored("How would you like to handle the missing data?\n"+
                                                "1 : Use Interpolate\n"+
                                                "2 : Drop Them (remove them from the data)\n"+
                                                ">>> ", colors["input"])))
                        
                        # Perform selected method to handle missing data
                        if how_ == 1:
                            ada.fill_na(how="interpolate")
                            os.system("cls || clear")
                            print(colored(f"Missing values after Interpolate() : {ada.df.isna().sum().sum()}", colors["output"]))

                        elif how_ == 2:
                            ada.fill_na(how="drop_na")
                            os.system("cls || clear")
                            print(colored(f"Missing values after drop_na() : {ada.df.isna().sum().sum()}", colors["output"]))

                        else:
                            raise SystemExit("Program exitted...")

                else:
                    print(colored(f"No missing values.", colors["success"]))
                    
                # Determine whether to use regressor or classifier based on target column type
                if ada.regressor_classifier_:
                    print(colored("Target column is continuous --> regressor will be used.", colors["output"]))
                    model = Regressor(X=ada.features, y=ada.target)
                    print(colored("Creating models:", colors["output"]))
                    model.create_models()

                else:
                    print(colored("Target column is categorical --> classifier will be used.", colors["output"]))
                    model = Classifier(X=ada.features, y=ada.target)
                    model.create_models()
                    os.system("cls || clear")
                    print(colored(f"Models created successfully", colors["success"]))

                print(colored(model.get_best_estimator(), colors["success"]))

                # Ask user whether to display scores for each model
                get_scores_ = input(colored("Would you like to have all scores for each model (y/n)?\n>>> ", colors["input"]))
                if get_scores_.lower() == "y":
                    os.system("cls || clear")
                    model.get_scores()

                # Ask user whether to display parameters for each model
                get_model_params_ = input(colored("Would you like to get the params for each model (y/n)?\n>>> ", colors["input"]))
                if get_model_params_.lower() == "y":
                    os.system("cls || clear")
                    for k, val in model.estimator_params.items():
                        print(colored(f"{k} : {val}\n", colors["output"]))

                # Ask user whether to proceed with the current results and create the final model
                create_final_model_ = input(colored("Would you like to proceed with these results and create the final model (y/n)?\n>>> ", colors["input"]))
                if create_final_model_:
                    os.system("cls || clear")
                    print(colored("Creating final model:", colors["output"]))
                    model.create_final_model()
                    print(colored("Final model created successfully", colors["success"]))

                else:
                    raise SystemExit("Program exitted...")

                # Ask user whether to dump the final model as a .joblib file
                dump_model_ = input(colored("Would you like to dump the model as a .joblib-file (y/n)?\n>>> ", colors["input"]))
                if dump_model_.lower() == "y":
                    file_name = self.open_file_explorer()
                    dump(model._final_model, file_name)
                    print(colored("Final model was successfully dumped", colors["success"]))

                else:
                    raise SystemExit("Program exitted...")
                
            except Exception as e:
                print(f"Error processing CSV file: {e}")

    # Method to open file explorer and let user save their final estimator
    def open_file_explorer(self):
        root = tk.Tk()
        root.withdraw()
        filepath = filedialog.asksaveasfilename(defaultextension=".joblib", filetypes=[("Joblib files", "*.joblib")])

        return filepath

# Main function to run the program
def main():
    root = tk.Tk()
    app = CSVSelector(root)
    root.mainloop()

# Standard procedure to call the main function
if __name__ == "__main__":
    main()
