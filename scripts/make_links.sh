#!/bin/bash

# This script creates symbolic links to the work_dirs, checkpoints, and tmp_ckpts directories
# Depending on the machine we are running on, it creates links to the appropriate directories

# Determine the machine we are running on
machine=$(hostname)


# Make sure we are running this script in the root directory of the project, called finetune-SAM2
current_dir=$(basename $(pwd))
if [ "$current_dir" != "finetune-SAM2" ]; then
    echo "Error: This script must be run from the finetune-SAM2 directory"
    exit 1
fi

# Check if files (work_dirs, checkpoints, tmp_ckpts) are symbolic links
if [ -L "work_dirs" ] || [ -L "checkpoints" ] || [ -L "tmp_ckpts" ] || [ -L "data" ]; then
    # Check if the links are broken
    if [ ! -e "work_dirs" ] || [ ! -e "checkpoints" ] || [ ! -e "tmp_ckpts" ] || [ ! -e "data" ]; then
        echo "One or more symbolic links are broken. Removing them..."
        rm -rf work_dirs checkpoints tmp_ckpts data
    else # They are not broken, so we don't need to do anything
        echo "Files are already symbolic links."
        exit 0
    fi
else
    echo "Files are not symbolic links."
fi

    
if [ "$machine" == "w8724.see.ed.ac.uk" ]; then # balduran
    ln -s /localdisk/data2/miguel/projects_storage/finetune-SAM2/work_dirs ./work_dirs
    ln -s /localdisk/data2/miguel/projects_storage/finetune-SAM2/checkpoints ./checkpoints
    ln -s /localdisk/data2/miguel/projects_storage/finetune-SAM2/tmp_ckpts ./tmp_ckpts    
    ln -s /localdisk/data2/miguel/datasets ./data
fi

if [ "$machine" == "w7830.see.ed.ac.uk" ]; then # a100
    ln -s /localdisk/data2/Users/s2254242/projects_storage/finetune-SAM2/work_dirs ./work_dirs
    ln -s /localdisk/data2/Users/s2254242/projects_storage/finetune-SAM2/checkpoints ./checkpoints
    ln -s /localdisk/data2/Users/s2254242/projects_storage/finetune-SAM2/tmp_ckpts ./tmp_ckpts
    ln -s /localdisk/data2/Users/s2254242/datasets ./data
fi

if [ "$machine" == "w8870.see.ed.ac.uk" ]; then # claptrap
    ln -s /localdisk/home/s2254242/projects_storage/finetune-SAM2/work_dirs ./work_dirs
    ln -s /localdisk/home/s2254242/projects_storage/finetune-SAM2/checkpoints ./checkpoints
    ln -s /localdisk/home/s2254242/projects_storage/finetune-SAM2/tmp_ckpts ./tmp_ckpts
    ln -s /localdisk/home/s2254242/datasets ./data
fi

echo "Links created successfully"