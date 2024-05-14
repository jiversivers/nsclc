#!/bin/bash

# Create a temporary job script to submit
tmp_job_script=$(mktemp)
echo "#!/bin/bash" > "$tmp_job_script"

# Append the output of the base_job into the temp script
./base_job.sh "$@" >> "$tmp_job_script"

# Make the temp script executable
chmod +x "$tmp_job_script"

# Submit the temp script
sbatch "$tmp_job_script"

# Clean up the temp script
rm "$tmp_job_script"