#! /bin/bash

SCRIPT_DIR=`realpath $(dirname "$0")`
GENERATE_RAVENS_SCRIPT_DIR=/cbica/home/giannoua/test_pipeline_istag/src

# Defaults
RavensReg=0.3
RavensScaleFactor=1000
method=ants
label=1
outdir=""

print_usage() {
  echo "Usage: $0 -s <subject_t1> -l <subject_seg> -t <template_t1> -o <out_dir> [-m ants|synthmorph] [-p ravens_reg] [-f scale] [-i label]"
}

while getopts ":s:l:t:o:m:p:f:i:h" opt; do
  case $opt in
    s) t1="$OPTARG" ;;
    l) t1seg="$OPTARG" ;;
    t) img_temp="$OPTARG" ;;
    o) outdir="$OPTARG" ;;
    m) method="$OPTARG" ;;
    p) RavensReg="$OPTARG" ;;
    f) RavensScaleFactor="$OPTARG" ;;
    i) label="$OPTARG" ;;
    h) print_usage; exit 0 ;;
    *) echo "Unknown option: -$OPTARG"; print_usage; exit 1 ;;
  esac
done

if [[ -z "$t1" || -z "$t1seg" || -z "$img_temp" || -z "$outdir" ]]; then
  echo "ERROR: Missing required arguments"; print_usage; exit 1
fi

mkdir -pv "$outdir"

bname=`basename ${t1}`
template_name=$(basename "$img_temp" | sed 's/\.nii\.gz$//' | sed 's/\.nii$//')
oWarp=${outdir}/${bname%.nii.gz}_warpedTo-${template_name}

cmd="sbatch \
  --output="$SCRIPT_DIR"/logs/ants-%j.out \
  --cpus-per-task=8 \
  --mem-per-cpu=4G \
  --time=06:00:00 \
  --propagate=NONE \
  --ntasks=1 \
  --verbose \
  ${GENERATE_RAVENS_SCRIPT_DIR}/GenerateRAVENS.sh -s $t1 -t ${img_temp} -o $oWarp -l ${t1seg} -p ${RavensReg} -i ${label} -f ${RavensScaleFactor} -m $method"

echo "About to run: $cmd"
eval $cmd
