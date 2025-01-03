#!/bin/bash

# 定义一个函数来处理文件
process_file() {
  local filename=$1

  # 检查文件是否存在
  if [ ! -f "$filename" ]; then
    echo "Error: File '$filename' not found."
    return
  fi

  # 定义要写入的内容的开始和结束标记
  begin_marker="### begin: $filename"
  end_marker="### end: $filename"

  # 检查 codes.txt 是否存在，如果不存在则创建
  if [ ! -f "codes.txt" ]; then
    touch codes.txt
  fi

  # 删除原有的标记和内容
  sed -i "/$(echo $begin_marker | sed 's/\//\\\//g')/,/$(echo $end_marker | sed 's/\//\\\//g')/d" codes.txt

  # 将新内容追加到 codes.txt 的末尾
  echo -e "$begin_marker" >>codes.txt
  cat "$filename" >>codes.txt
  echo -e "$end_marker" >>codes.txt

  echo "Content from '$filename' has been appended to codes.txt"
}

# 调用函数处理输入的文件或文件夹

# 检查是否提供了文件或文件夹参数
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <file_or_directory>"
  exit 1
fi

# 获取文件或文件夹参数
input=$1

# 检查输入的是一个文件还是文件夹
if [ -f "$input" ]; then
  # 如果是文件，直接处理这个文件
  process_file "$input"
elif [ -d "$input" ]; then
  # 如果是文件夹，处理文件夹中的所有文件
  # find "$input" -type f | while read -r file; do
  find "$input" -type f \
    -not -path '*/node_modules/*' \
    -not -path '*\.git/*' \
    -not -path '*\.md' \
    -not -path '*\.cache/*' \
    -not -path '*\.svg' \
    -not -path '*\.yaml' \
    -not -path '*config*' \
    -not -path '*package*' \
    -not -path '*env*' \
    -not -path '*/data*' \
    -not -path '*\.gitignore' \
    -not -path '*migrations/*' \
    -not -path '*__pycache__/*' \
    -not -path '*\.sqlite3' \
    -not -path '*__init__\.py' \
    -not -path '*codes\.txt' \
    -not -path '*/build*' \
    -not -path '*\.png' |
    while read -r file; do
      process_file "$file"
    done
else
  echo "Error: '$input' is neither a file nor a directory."
  exit 1
fi
