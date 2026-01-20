#!/bin/bash

# This script creates symbolic links to the work_dirs, checkpoints, and tmp_ckpts directories
# Depending on the machine we are running on, it creates links to the appropriate directories

# Determine the machine we are running on
machine=$(hostname)

# List of folders to check
folders=("work_dirs" "checkpoints" "tmp_ckpts" "data" "results_analysis")

# Check each folder
for folder in "${folders[@]}"; do
    if [ -L "$folder" ]; then
        # It is a symbolic link
        if [ -e "$folder" ]; then
            # Link is not broken
            echo "$folder: Link is not broken, doing nothing."
        else
            # Link is broken
            echo "$folder: Link is broken, deleting..."
            rm "$folder"
            echo "$folder: Link deleted successfully."
        fi
    else
        # It is not a symbolic link
        if [ -e "$folder" ]; then
            echo "$folder: Not a symbolic link (regular folder exists), cannot delete."
        else
            echo "$folder: Does not exist, will create link."
        fi
    fi
done

# Create links based on machine
if [ "$machine" == "w8724.see.ed.ac.uk" ]; then # balduran
    ln -s /localdisk/data2/miguel/projects_storage/finetune-SAM2/work_dirs ./work_dirs
    ln -s /localdisk/data2/miguel/projects_storage/finetune-SAM2/checkpoints ./checkpoints
    ln -s /localdisk/data2/miguel/projects_storage/finetune-SAM2/tmp_ckpts ./tmp_ckpts    
    ln -s /localdisk/data2/miguel/datasets ./data
    ln -s /localdisk/data2/miguel/projects_storage/finetune-SAM2/results_analysis ./results_analysis
fi

if [ "$machine" == "w7830.see.ed.ac.uk" ]; then # a100
    ln -s /localdisk/data2/Users/s2254242/projects_storage/finetune-SAM2/work_dirs ./work_dirs
    ln -s /localdisk/data2/Users/s2254242/projects_storage/finetune-SAM2/checkpoints ./checkpoints
    ln -s /localdisk/data2/Users/s2254242/projects_storage/finetune-SAM2/tmp_ckpts ./tmp_ckpts
    ln -s /localdisk/data2/Users/s2254242/datasets ./data
    ln -s /localdisk/data2/Users/s2254242/projects_storage/finetune-SAM2/results_analysis ./results_analysis
fi

if [ "$machine" == "w8870.see.ed.ac.uk" ]; then # claptrap
    ln -s /localdisk/home/s2254242/projects_storage/finetune-SAM2/work_dirs ./work_dirs
    ln -s /localdisk/home/s2254242/projects_storage/finetune-SAM2/checkpoints ./checkpoints
    ln -s /localdisk/home/s2254242/projects_storage/finetune-SAM2/tmp_ckpts ./tmp_ckpts
    ln -s /localdisk/home/s2254242/datasets ./data
fi

echo "Links created successfully"