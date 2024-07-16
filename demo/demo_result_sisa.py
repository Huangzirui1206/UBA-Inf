import os
import sys
import argparse
import pandas as pd

# TODO: change this as you want, say perturb_badnet_poionly
result_folder_path = "./record/your_sisa_result_folder_name"
# You can directly set this in command line with argument --result_folder_path

"""
This read the final BA and ASR in the SISA result folder
This model must be trained with SISA, with folder structure as:
- your_result_folder_name
|-- shard0      # One shard in SISA
    |-- cover          # pre-unlearning models 
        |-- acc_like_metric_plots.png # the plot of accuracy metrics of pre-unlearning models
        |-- cover_df.csv              # the training metrics after each epoch
        |-- cover_df_summary.csv      # the summary of final metrics of pre-unlearning models
        |-- cover_result.pt           # the model parameter of pre-unlearning models
        |-- loss_metrc_plots.png      # the plot of loss metrics of pre-unlearning models
    |-- unlearn        # post-unlearning models 
        |-- acc_like_metric_plots.png # the plot of accuracy metrics of post-unlearning models
        |-- unlearn_df.csv            # the training metrics after each epoch
        |-- unlearn_df_summary.csv    # the summary of final metrics of post-unlearning models
        |-- unlearn_result.pt         # the model parameter of post-unlearning models
        |-- loss_metrc_plots.png      # the plot of loss metrics of pre-unlearning models
|-- shard1
    |-- cover
        |-- ...
    |-- unlearn
        |-- ...
|-- ...     # more shard folders
|-- ...
|-- 20xx_xx_xx_xx_xx_xx.log           # the log file are named after operation time
|-- cover_summary.csv                 # The aggregate summary of pre-unlearning SISA model
|-- info.pickle                       # information
|-- unlearn_summary.csv               # The aggregate summary of post-unlearning SISA model
             
If you want to check the final results directly, you can just turn to cover_summary.csv and unlearn_summary.csv.
cover_summary.csv repersents the pre-unlearning SISA model, while unlearn_summary.csv represents the post-unlearning SISA model.
More precisely, you should look at the "test_acc" and "test_asr" rows.

This demo file will read the results in cover_summary.csv and unlearn_summary.csv and directly print the results on command terminal.
"""

parser = argparse.ArgumentParser()

parser.add_argument("--result_folder_path", type=str,
                    default="./record/your_result_folder_name",
                    help="You can directly set this in command line with argument --result_folder_path.")
args = parser.parse_args()    

cover_df = pd.read_csv(f"{args.result_folder_path}/cover_summary.csv")
unlearn_df = pd.read_csv(f"{args.result_folder_path}/unlearn_summary.csv")

cover_result = cover_df.values
cover_acc = cover_result[1,1]
cover_asr = cover_result[2,1]

unlearn_result = unlearn_df.values
unlearn_acc = unlearn_result[1,1]
unlearn_asr = unlearn_result[2,1]

print(f"Before unlearning, the BA is {cover_acc} while ASR is {cover_asr}.")
print(f"After unlearning, the BA is {unlearn_acc} while ASR is {unlearn_asr}.")
