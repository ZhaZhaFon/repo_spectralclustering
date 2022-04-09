

infer:
	CUDA_VISIBLE_DEVICES=0 python infer_rttm.py --input_audio /home/zzf/dataset/voxconverse/test/aepyx.wav --output_path .
eval:
# 	https://github.com/ZhaZhaFon/repo_VoxSRC20-evaltoolkits
	python ../repo_VoxSRC20-evaltoolkits/validate_rttm.py ./aepyx.rttm
	python ../repo_VoxSRC20-evaltoolkits/compute_diarisation_metrics.py -r /home/zzf/codebase/repo_VoxSRC20-evaltoolkits/voxconverse/*.rttm -s ./*.rttm

infer_all:
	CUDA_VISIBLE_DEVICES=0 python infer_all.py --input_dir /home/zzf/dataset/voxconverse/test --output_dir /home/zzf/experiment-sd/spectralclustering_0409
eval_all:
# 	https://github.com/ZhaZhaFon/repo_VoxSRC20-evaltoolkits
	python ../repo_VoxSRC20-evaltoolkits/compute_diarisation_metrics.py -r /home/zzf/dataset/voxconverse/voxconverse_label/test/*.rttm -s /home/zzf/experiment-sd/spectralclustering_0409/output_rttm/*.rttm