#! /bin/bash

directory=${1}

for subdir in "${directory}"/*/
do
     dirbase=$(basename "$subdir")
     dirname="${dirbase%.*}" 
   
     if [[ "${dirname}" == *cov ]]; then
          python ./defense/clp.py --result_file ${dirname} --yaml_path ./config/defense/clp/gtsrb.yaml --dataset gtsrb
          python ./defense/i-bau.py --result_file ${dirname} --yaml_path ./config/defense/i-bau/gtsrb.yaml --dataset gtsrb
          python ./defense/nc.py --result_file ${dirname} --yaml_path ./config/defense/nc/gtsrb.yaml --dataset gtsrb --epochs 10
          python ./defense/anp.py --result_file ${dirname} --yaml_path ./config/defense/anp/gtsrb.yaml --dataset gtsrb
          python ./defense/nad.py --result_file ${dirname} --yaml_path ./config/defense/nad/gtsrb.yaml --dataset gtsrb --epochs 10
          python ./defense/fp.py --result_file ${dirname} --yaml_path ./config/defense/fp/gtsrb.yaml --dataset gtsrb --epochs 10
     fi
done
