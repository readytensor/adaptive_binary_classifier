#!/bin/bash
# fix_line_endings.sh - Script to convert line endings of Python scripts from Windows to Unix format

# This script uses the 'dos2unix' utility to convert line endings in Python script files (.py) from Windows-style (CRLF) to Unix-style (LF).
# It is intended to fix line ending issues that can occur when scripts created on Windows machines are used in Unix-like environments.

# Instructions:
# 1. Place this script in the root directory of your repository (directory which contains the `src` directory).
# 2. Ensure that the 'dos2unix' utility is installed in the Docker container (it is used to perform the line ending conversion).
#    You can install it by adding 'RUN apt-get update && apt-get install -y dos2unix' to your Dockerfile.
# 3. Make this script executable by running 'chmod +x fix_line_endings.sh'.
# 4. During container startup, this script will automatically convert the line endings of all Python script files within the specified directory.

# Set the path to the directory containing your Python script files
SCRIPTS_DIR="/opt/src"

# Use the 'find' command to locate Python script files in the specified directory and its subdirectories
# Then, execute 'dos2unix' on each file to convert line endings
find "$SCRIPTS_DIR" -type f -name '*.py' -exec dos2unix {} \;

# Note: This script assumes that the 'dos2unix' utility is available in the system path.
# If you encounter any issues, ensure that 'dos2unix' is properly installed and accessible.

# End of script
