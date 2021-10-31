#!/bin/bash

SECONDS=0


data_dir=/app/haozhe/OmniPrint-metaX_ImageFolder

METHOD="DAAN"

python main.py --method_name "${METHOD}" --config "${METHOD}/${METHOD}.yaml" --data_dir $data_dir --src_domain meta1 --tgt_domain meta2 | tee DAAN_1to2.log


DURATION=$SECONDS

echo "End of the program! $(($DURATION / 60)) minutes and $(($DURATION % 60)) seconds elapsed." 

