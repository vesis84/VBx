#!/bin/bash

# Run VBx diarization on a single wav+vad file-pair, produce RTTM.

# KarelV: it does not work well,
# - it is splitting some pilot's speech, but not distinguishing the Pilot from ATCOs

# Possible next steps:
# - we could re-train the PLDA? (with few hours of data)
# - or re-train the x-vector extractor (trained on voxceleb)

set -euxo pipefail

[ $# -ne 3 ] && echo "$0 <audio_in> <vad_segs_in> <rttm_out>" && exit 1
AUDIO_IN=$1
VAD_SEGS_IN=$2
RTTM_OUT=$3

# tmp folder for intermediate files,
tmp=$(mktemp -d); # trap "rm -r $tmp" EXIT # clean-up
diarization_tmp=$tmp/diarization; mkdir -p $diarization_tmp

# $dir is a location of this script...
dir=$(readlink -m $(dirname $0))
model_init=$dir/xvector_extractor.txt

# get basename and ext of 'AUDIO_IN',
filename="${AUDIO_IN##*/}" # filename.wav
format="${filename##*.}"   # wav
name="${filename%.*}"      # filename

# Copy the VAD locally, make sure the name is same as for 'AUDIO_IN',
# (format conversion, '<utt> <rec> <t_beg> <t_end>' -> '<t_beg> <t_end> sp')
mkdir -p ${diarization_tmp}/vad
awk '{ print $3, $4, "sp"; }' <${VAD_SEGS_IN} \
  >${diarization_tmp}/vad/${name}.lab

# folders with data,
dir_audio=$(dirname $AUDIO_IN)
dir_vad=${diarization_tmp}/vad

# create a list file with 1 record,
list_file=$diarization_tmp/list
echo "$name=$name" > $list_file

$dir/extract.py --in-file-list $list_file \
                --in-vad-dir $dir_vad \
                --in-audio-dir $dir_audio \
                --in-format $format \
                --ark-file $diarization_tmp/xvectors.ark \
                --scp-file $diarization_tmp/xvectors.scp \
                --segment-file $diarization_tmp/segments \
                --model-init $model_init

alpha=0.55
thr=0.0
tareng=0.3
smooth=5.0
lda_dim=220
Fa=0.4
Fb=11
loopP=0.80

# x-vector clustering using VBHMM based diarization
$dir/diarization_PLDAadapt_AHCxvec_BHMMxvec.py \
                    $diarization_tmp \
                    $diarization_tmp/xvectors.ark \
                    $diarization_tmp/segments \
                    $dir/mean.vec \
                    $dir/transform.mat \
                    $dir/plda_voxceleb \
                    $dir/plda_dihard \
                    $alpha \
                    $thr \
                    $tareng \
                    $smooth \
                    $lda_dim \
                    $Fa \
                    $Fb \
                    $loopP

# export the output RTTM,
cp ${diarization_tmp}/${name}.rttm ${RTTM_OUT}

exit 0 # Success!
