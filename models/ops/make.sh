#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

set -euo pipefail

# ---------------------------------------------------------------------------
# Workaround for CUDA 12.8 + glibc >= 2.40:
#   math_functions.h declares cospi/sinpi/cospif/sinpif without noexcept,
#   which clashes with glibc's noexcept(true) declarations.
#   Fix from: https://forums.developer.nvidia.com/t/323591/3
#
# Since we can't write to the shared HPC CUDA module, we create a local
# symlinked overlay of CUDA_HOME with only math_functions.h patched,
# then redirect CUDA_HOME so nvcc picks up the fix through its own
# internal include resolution.
# ---------------------------------------------------------------------------
CUDA_HOME="${CUDA_HOME:-$(python -c 'from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)')}"
PATCHED_CUDA="$(pwd)/cuda_patched"
ORIG_HEADER="${CUDA_HOME}/include/crt/math_functions.h"
PATCHED_HEADER="${PATCHED_CUDA}/include/crt/math_functions.h"

if [ -f "$ORIG_HEADER" ] && [ ! -f "$PATCHED_HEADER" ]; then
    echo "Creating patched CUDA overlay at ${PATCHED_CUDA}"
    rm -rf "$PATCHED_CUDA"
    mkdir -p "$PATCHED_CUDA"

    # Symlink all top-level entries (bin, lib64, nvvm, etc.)
    for item in "$CUDA_HOME"/*; do
        ln -sf "$item" "$PATCHED_CUDA/$(basename "$item")"
    done

    # Replace the include symlink with a real directory of symlinks
    rm -f "$PATCHED_CUDA/include"
    mkdir -p "$PATCHED_CUDA/include"
    for item in "$CUDA_HOME"/include/*; do
        ln -sf "$item" "$PATCHED_CUDA/include/$(basename "$item")"
    done

    # Replace the crt symlink with a real directory of symlinks
    rm -f "$PATCHED_CUDA/include/crt"
    mkdir -p "$PATCHED_CUDA/include/crt"
    for item in "$CUDA_HOME"/include/crt/*; do
        ln -sf "$item" "$PATCHED_CUDA/include/crt/$(basename "$item")"
    done

    # Copy and patch math_functions.h (the only file we actually modify)
    rm -f "$PATCHED_HEADER"
    cp "$ORIG_HEADER" "$PATCHED_HEADER"
    sed -i -E 's/^(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ (double|float) +((sin|cos)pif?) *\([^)]+\)) *;/\1 noexcept (true);/' \
        "$PATCHED_HEADER"

    echo "Patch applied."
else
    echo "Using existing patched CUDA overlay at ${PATCHED_CUDA}"
fi

export CUDA_HOME="$PATCHED_CUDA"

FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" \
    python setup.py build install
