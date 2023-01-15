#!/bin/bash -eux

function fail {
    echo "TEST FAILED: $1"
    exit 1
}

function assert_file_exists {
    filename=$1
    if [ -f "$filename" ]; then
        echo "OK"
    else
        fail "File does not exist: $filename"
    fi
}

rm -f output.0.png
cog predict -i prompt="grazing alien" -i negative_prompt="sheep" -i image=@desktop.jpg -i mask=@desktop-mask.jpg -i seed=1
assert_file_exists output.0.png
rm output.0.png

cog predict -i prompt="grazing alien" -i negative_prompt="sheep" -i image=@desktop.jpg -i mask=@desktop-mask.jpg -i invert_mask=true -i seed=1
assert_file_exists output.0.png
rm output.0.png

cog push
