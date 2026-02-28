#!/bin/bash

TARGET_DIR="../../../../cgRX_March2025/rtx-index/index-prototype/src/"

# Copy files with relevant extensions to target dir
for ext in cuh h cpp cu sh; do
    for file in *.$ext; do
        [ -e "$file" ] && cp "$file" "$TARGET_DIR"
    done
done

cd "$TARGET_DIR" || exit 1

git add .
git commit -m "$1"
git push origin main
