INSTRUCTION=$1 # xvectors or VBx or score
SET=$2 # dev or eval
DIHARD_DIR=$3 # directory containing the DIHARD data as provided by the organizers

if [[ $SET = "dev" ]]; then
	DATA_DIR=$DIHARD_DIR/LDC2019E31_Second_DIHARD_Challenge_Development_Data
elif [[ $SET = "eval" ]]; then
	DATA_DIR=$DIHARD_DIR/LDC2019E32_Second_DIHARD_Challenge_Evaluation_Data_V1.1
else
	echo "The set has to be 'dev' or 'eval'"
	exit -1
fi

TMP_DIR=./2tmp_dir_$SET
OUT_DIR=./2out_dir_$SET

mkdir -p $TMP_DIR $OUT_DIR 


if [[ $INSTRUCTION = "xvectors" ]]; then
	model_init=xvector_extractor.txt
	# - extract speech segment according to VAD
	# - each speech segment split into 1.5s subsegments with shift 0.25s
	./extract.py --in-file-list list_$SET \
				 --in-vad-dir $DATA_DIR/data/single_channel/sad \
				 --in-audio-dir $DATA_DIR/data/single_channel/flac \
				 --in-format flac \
				 --ark-file $TMP_DIR/xvectors.ark \
				 --scp-file $TMP_DIR/xvectors.scp \
				 --segment-file $TMP_DIR/segments \
                 --model-init $model_init
fi


if [[ $INSTRUCTION = "VBx" ]]; then
	alpha=0.55
	thr=0.0
	tareng=0.3
	smooth=5.0
	lda_dim=220
	Fa=0.4
	Fb=11
	loopP=0.80

	# x-vector clustering using VBHMM based diarization
	./diarization_PLDAadapt_AHCxvec_BHMMxvec.py \
	 					$OUT_DIR \
	 					$TMP_DIR/xvectors.ark \
	 					$TMP_DIR/segments \
	 					mean.vec \
	 					transform.mat \
	 					plda_voxceleb \
	 					plda_dihard \
	 					$alpha \
	 					$thr \
	 					$tareng \
	 					$smooth \
	 					$lda_dim \
	 					$Fa \
	 					$Fb \
	 					$loopP
fi


if [[ $INSTRUCTION = "score" ]]; then
	SCORE_DIR=$4 # directory with scoring tool: https://github.com/nryant/dscore
	python $SCORE_DIR/score.py \
		--collar 0.0 \
		-u $DATA_DIR/data/single_channel/uem/all.uem \
		-r $DATA_DIR/data/single_channel/rttm/*.rttm \
		-s $OUT_DIR/*.rttm
fi