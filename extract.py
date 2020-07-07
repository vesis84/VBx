#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# @Authors: Shuai Wang, Federico Landini
# @Emails: wsstriving@gmail.com, landini@fit.vutbr.cz

import torch
import sys, os
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import random
import numpy as np
import kaldi_io
from tdnn_model import load_kaldi_model
import features
import soundfile as sf
from scipy.io import wavfile

torch.backends.cudnn.benchmark=True


def validate_path(dir_name):
    """
    :param dir_name: Create the directory if it doesn't exist
    :return: None
    """
    dir_name = os.path.dirname(dir_name)  # get the path
    if not os.path.exists(dir_name) and (dir_name is not ''):
        os.makedirs(dir_name)


class features_generator():
    def __init__(self, in_file_list, in_vad_dir, in_audio_dir, in_format):
        file_names = np.loadtxt(in_file_list, dtype=object).reshape((-1,1))
        self.file_names = [f[0].split('=')[-1] for f in file_names]
        self.in_vad_dir = in_vad_dir
        self.in_audio_dir = in_audio_dir
        self.in_format = in_format

    def __iter__(self):
        return self

    def __next__(self):
        noverlap = 240
        winlen = 400
        fs = 16000
        window = features.povey_window(winlen)
        fbank_mx = features.mel_fbank_mx(winlen, fs, NUMCHANS=40, LOFREQ=20.0, HIFREQ=7600, htk_bug=False)
        LC = 150
        RC = 149
        step = 150
        shift = 25
        min_length = 10

        for fn in self.file_names:
            print(fn)
            if os.path.isfile(self.in_vad_dir+"/"+fn+".lab"):
                labs = (np.loadtxt(self.in_vad_dir+"/"+fn+".lab", usecols=(0,1))*16000).astype(int)
            else:
                sys.exit("VAD segmentation extension must be .lab")
            if self.in_format == 'flac':
                signal, samplerate = sf.read(self.in_audio_dir+"/"+fn+".flac")
            elif self.in_format == 'wav':
                signal, samplerate = wavfile.read(self.in_audio_dir+"/"+fn+".wav")
            else:
                sys.exit("The audio input must be .flac or .wav")
            signal = features.add_dither((signal*2**(samplerate/1000 - 1)).astype(int))
            for segnum in range(len(labs)):
                seg=signal[labs[segnum,0]:labs[segnum,1]]
                seg=np.r_[seg[noverlap//2-1::-1], seg, seg[-1:-winlen//2-1:-1]] # Mirror noverlap//2 initial and final samples
                fea = features.fbank_htk(seg, window, noverlap, fbank_mx, USEPOWER=True, ZMEANSOURCE=True)
                fea = features.cmvn_floating_kaldi(fea, LC, RC, norm_vars=False)
                slen = len(fea)
                start=-shift
                for start in range(0,slen-step,shift):
                    name = "%s_%04d-%08d-%08d" % (fn, segnum, start, (start+step))
                    feat = fea[start:start+step]
                    segment = "%s_%04d-%08d-%08d %s %g %g\n" % (fn, segnum, start, (start+step), fn, labs[segnum,0]/float(fs)+start/100.0, labs[segnum,0]/float(fs)+(start+step)/100.0)
                    yield name, feat, segment
                if slen-start-shift > min_length:
                    start += shift
                    name = "%s_%04d-%08d-%08d" % (fn, segnum, start, slen)
                    feat = fea[start:slen]
                    segment = "%s_%04d-%08d-%08d %s %g %g\n" % (fn, segnum, start, slen, fn, labs[segnum,0]/float(fs)+start/100.0, labs[segnum,0]/float(fs)+start/100.0+(slen-start)/100.0)
                    yield name, feat, segment


def write_ark(model, args):
    """
    Write the extracted embeddings to ark file, having the same format as i-vectors
    """
    gen = features_generator(args.in_file_list, args.in_vad_dir, args.in_audio_dir, args.in_format)
    model.eval()
    save_to = args.ark_file
    validate_path(save_to)

    with torch.no_grad():
        ark_scp_output = 'ark:| copy-vector ark:- ark,scp:'+args.ark_file+','+args.scp_file
        with open(args.segment_file, "w") as seg_file:
            with kaldi_io.open_or_fd(ark_scp_output, "wb") as ark_scp_file:
                for name, data, seg in next(gen):
                    data = np.reshape(data, (1, data.shape[0], data.shape[1]))
                    data = torch.from_numpy(data).to(args.cuda, dtype=torch.double)
                    data = data.transpose(1,2)
                    _, embedding_a, embedding_b = model(data)
                    vector = embedding_a.data.cpu().numpy()
                    kaldi_io.write_vec_flt(ark_scp_file, vector[0], name)
                    seg_file.write(seg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file-list', type=str, help="Input scp list")
    parser.add_argument('--in-vad-dir', type=str, help="Input VAD dir")
    parser.add_argument('--in-audio-dir', type=str, help="Input audio dir")
    parser.add_argument('--in-format', type=str, help="Input format")
    parser.add_argument('--ark-file', type=str, help="Output embeddings ark")
    parser.add_argument('--scp-file', type=str, help="Output embeddings scp")
    parser.add_argument('--segment-file', type=str, help="Output embeddings segments")
    parser.add_argument("--model-init", type=str, default=None)

    args = parser.parse_args()

    if args.model_init is not None:
        model = load_kaldi_model(args.model_init)
    print(model)

    args.cuda = torch.device("cpu")
    #args.cuda = torch.device("cuda")
    #os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    model = model.to(args.cuda, dtype=torch.double)
    write_ark(model, args)


if __name__ == '__main__':
    main()
