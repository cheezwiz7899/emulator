#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compile the real GC request flow with a fake Vulkan allocator, without a GPU."""
import argparse
import pathlib
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("output", type=pathlib.Path)
parser.add_argument("--revision", help="Optionally check the unmodified flow at a Git revision")
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parents[2]


def read(path):
    if args.revision:
        return subprocess.check_output(["git", "show", f"{args.revision}:{path}"], cwd=root,
                                       text=True)
    return (root / path).read_text()


def definition(text, signature):
    start = text.index(signature)
    opening = text.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


header = read("src/video_core/renderer_vulkan/vk_staging_buffer_pool.h")
source = read("src/video_core/renderer_vulkan/vk_staging_buffer_pool.cpp")
payload = definition(header, "struct StagingBufferRef") + ";"
declaration = re.search(r"([^;\n]+\bRequestGCDownload\(size_t size\));", header).group(1)
member = re.search(r"([^;\n]+\bgc_download);", header).group(1) + ";"
alias = re.search(r"using GCDownloadRef = [^;]+;", header)
mutex = "std::mutex gc_download_mutex;" if "std::mutex gc_download_mutex;" in header else ""
signature = re.search(r"[^\n]+StagingBufferPool::RequestGCDownload\(size_t size\)", source).group(0)
args.output.write_text(
    payload + "\n" + (alias.group(0) if alias else "") + "\n"
    + "class StagingBufferPool {\npublic:\n" + declaration + ";\n" + member + "\n" + mutex
    + "\n#include \"fake_allocator.inc\"\n};\n" + definition(source, signature) + "\n")
