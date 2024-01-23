#!/bin/bash
PROJ_DIR="$(cd "$(dirname "${0}")" && pwd)"

if [ ! -d "${PROJ_DIR}/ckpts" ]; then
    mkdir ${PROJ_DIR}/ckpts
fi

# buggy planes offered by NVLab
# if [ ! -f "${PROJ_DIR}/ckpts/eg3d_1.zip" ]; then
#     wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/zip -O ${PROJ_DIR}/ckpts/eg3d_1.zip
# fi

# if [ -z "$(ls ${PROJ_DIR}/ckpts/*.pkl)" ]; then
#     unzip ${PROJ_DIR}/ckpts/eg3d_1.zip -d ckpts
# fi

echo "Please manually download ckpt files from https://drive.google.com/drive/folders/1eJrXvda9ZwA8NYOLtvr4N-iJ1u9wZ27J (Offered by https://github.com/oneThousand1000/LPFF-dataset/tree/master/networks)



