import subprocess

def run_multiple_scripts(script_name, base_args, ckpt_experiment_pairs, num_repeats):
    try:
        for ckpt_id, experiment_name in ckpt_experiment_pairs:
            # Update arguments with the current ckpt_id and experiment_name
            ckpt_id_arg = "None" if ckpt_id is None else ckpt_id
            args = base_args + ["--ckpt_id", ckpt_id_arg, "--experiment_name", experiment_name]
            
            # Run the script multiple times with the same arguments
            for _ in range(num_repeats):
                subprocess.run(['python', script_name] + args)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Define the script to run
    script_name = 'run_training_KD_h5.py'
    
    # Base arguments (common to all runs, except experiment name and ckpt_id)
    base_args = ["--subset", "5", "--n_epochs", "150" ,"--dir_prob", "0.6", "--mixstyle_p", "0.4", "--sample_rate", "32000"]
    
    # List of tuples containing checkpoint IDs and their corresponding experiment names
    ckpt_experiment_pairs = [
        
        (None, "NTU_KD_32K_FMS_DIR")          #Ptau
    ]
    
    # Number of times to repeat each experiment
    num_repeats = 5

    # Run the script with different checkpoint IDs and experiment names
    run_multiple_scripts(script_name, base_args, ckpt_experiment_pairs, num_repeats)
    