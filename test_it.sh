#!/bin/bash

WAV=/mnt/matylda5/iveselyk/ATCO2/GITLAB_usaar_atco2/kaldi_data_preparations/LIVEATC_BUT_v1/data_for_fede/UTC-20200306-14:27:56_LKPR_Approach.wav
VAD=/mnt/matylda5/iveselyk/ATCO2/GITLAB_usaar_atco2/kaldi_data_preparations/LIVEATC_BUT_v1/data_for_fede/UTC-20200306-14:27:56_LKPR_Approach.seg
RTTM=UTC-20200306-14:27:56_LKPR_Approach.rttm
./run_diarization.sh $WAV $VAD $RTTM


