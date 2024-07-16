import os
import sys
import argparse
import pandas as pd

# TODO: change this as you want, say perturb_badnet_poionly
result_folder_path = "./record/your_result_folder_name"
# You can directly set this in command line with argument --result_folder_path

"""
This read the final BA and ASR in the result folder
This model must be normally trained, with folder structure as:
- your_result_folder_name
|-- 20xx_xx_xx_xx_xx_xx.log         # the log file are named after operation time
|-- acc_like_metric_plots.png       # the plot of accuracy metrics during training
|-- attack_df.csv                   # the training metrics after each epoch
|-- attack_df_summary.csv           # the summary of final metrics
|-- cv_bd_index.pt                  # the indexes of camouflage and backdoor samples
|-- info.pickle                     # information
|-- loss_metric_plots.png           # the plot of loss metrics during training
|-- perturb_result.pt               # the model parameter

If you want to check the final results directly, you can just turn to attack_df_summary.csv.
More precisely, you should look at the "test_acc" and "test_asr" column in "last" row.

This demo file will read the results in attack_df_summay.csv and directly print the results on command terminal.
"""

parser = argparse.ArgumentParser()

parser.add_argument("--result_folder_path", type=str,
                    default="./record/your_result_folder_name",
                    help="You can directly set this in command line with argument --result_folder_path.")
args = parser.parse_args()    

df = pd.read_csv(f"{args.result_folder_path}/attack_df_summary.csv", usecols=['test_acc', 'test_asr'], nrows=1)

test_acc = df.at[0, 'test_acc']
test_asr = df.at[0, 'test_asr'] 

print(f"The BA is {test_acc}, while the ASR is {test_asr}.")
