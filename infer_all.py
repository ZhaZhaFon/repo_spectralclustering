import torchaudio as ta
import torch
import sys
import numpy as np
import os
import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', required=True, type=str, help='directory to input audio')
# https://pytorch.org/hub/snakers4_silero-vad_vad/
parser.add_argument('--pretrained_silerovad', required=False, type=str, default='/home/zzf/.cache/torch/hub/snakers4_silero-vad_master', help='path to sileroVAD pretrained model')
# https://github.com/ZhaZhaFon/repo_dvector
parser.add_argument('--repo_dvector', required=False, type=str, default='../repo_dvector', help='path to repo_dvector')
# https://github.com/ZhaZhaFon/repo_dvector/releases/tag/pretrained
parser.add_argument('--pretrained_dvector', required=False, type=str, default='/home/zzf/experiment-dvector/dvec_pool-attn/checkpoints/dvector-epoch299.pt', help='path to pretrained dvector model')
parser.add_argument('--output_dir', required=True, type=str, help='directory to write the output .rttm file')

def infer(args):
    
    
    print('')
    write_dir = os.path.join(args.output_dir, 'output_rttm')
    os.makedirs(write_dir, exist_ok=True)
    print(f'# 结果保存到{args.output_dir}')
    
    print(f'# 准备...')
    print('   # 准备VAD...')
    model, utils = torch.hub.load(repo_or_dir=args.pretrained_silerovad,
                                    model='silero_vad',
                                    force_reload=True,
                                    source='local')
    model = model.cuda()
    (get_speech_timestamps,
    save_audio,
    read_audio,
    VADIterator,
    collect_chunks) = utils
    print(f'   # 准备dvector...')
    sys.path.append(args.repo_dvector)
    import modules
    import data
    dvector = modules.AttentivePooledLSTMDvector(
            dim_input=40,
            seg_len=10,
            )
    ckpt = torch.load(args.pretrained_dvector)
    dvector.load_state_dict(ckpt.state_dict())
    dvector = dvector.cuda()
    mel_encoder = data.Wav2Mel()
    print(f'   # 准备spectralcluster...')
    import spectralcluster
    print(f'   # 准备VoxSRC-20...')
    
    
    print('')
    print('### START INFERENCE ###')
    print('')
    
    filelist = os.listdir(args.input_dir)
    for file in tqdm.tqdm(filelist):
        
        file_path = os.path.join(args.input_dir, file)
        print(f'# 处理{file_path}')
    
        wav, fs = ta.load(file_path)
        assert fs == 16000
        wav = wav.cuda()
        
        #print(f'   # VAD...')
        speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=fs)
    
        #print('   # dvector...')
        embs = []
        for segment_info in speech_timestamps:
            start = segment_info['start']
            end = segment_info['end']
            segment = wav[0][start:end].reshape(1, -1)
            #display(Audio(data=segment, rate=fs))
            mel = mel_encoder(segment.cpu(), sample_rate=fs).unsqueeze(0)
            embedding = dvector.embed_utterances(mel.cuda())
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
            embs.append(embedding.squeeze().cpu().numpy())
        embs = np.array(embs)
    
        #print('   # spectral clustering...')
        try:
            labels = spectralcluster.configs.icassp2018_clusterer.predict(embs)
        except:
            continue
        else:
            pass
        
        name = file_path.split('/')[-1].split('.')[0]
        write_path = os.path.join(write_dir, name+'.rttm')
        #print(f'   # writing .rttm file to {write_path}...')
        with open(write_path, 'w') as f:
            for (segment_info, label) in zip(speech_timestamps, labels):
                start = segment_info['start']
                end = segment_info['end']
                duration = end - start
                
                line = f'SPEAKER {name} 1 {start/fs} {duration/fs} <NA> <NA> spk{label} <NA> <NA>\n'
                f.write(line)
    
    print('')
    print('### INFERENCE COMPLETED ###')
    print('')
    print('done.')
    
if __name__ == '__main__':
    
    args = parser.parse_args()
    infer(args)