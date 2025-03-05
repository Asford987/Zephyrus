#!/bin/bash

# Find the project root (assumes a git repo or stops at /)
find_project_root() {
    local dir=$(pwd)
    while [ "$dir" != "/" ]; do
        if [ -d "$dir/.git" ]; then
            echo "$dir"
            return
        fi
        dir=$(dirname "$dir")
    done
    echo "Could not determine project root. Are you in a project?" >&2
    exit 1
}

# Find the build directory
find_build_dir() {
    local root_dir=$1
    local build_dir=$(find "$root_dir" -type d -name "build" -print -quit)
    if [ -z "$build_dir" ]; then
        echo "No build directory found in project." >&2
        exit 1
    fi
    echo "$build_dir"
}

# Main script logic
main() {
    project_root=$(find_project_root)
    build_dir=$(find_build_dir "$project_root")

    echo "Found build directory: $build_dir"
    rm -rf "$build_dir"/*
    echo "Build directory cleared: $build_dir"
}

main