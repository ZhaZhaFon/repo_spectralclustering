

infer:
	CUDA_VISIBLE_DEVICES=0 python infer_rttm.py --input_audio /home/zzf/dataset/voxconverse/test/aepyx.wav --output_path .
eval:
	python ../repo_VoxSRC20-evaltoolkits/validate_rttm.py ./aepyx.rttm
	python ../repo_VoxSRC20-evaltoolkits/compute_diarisation_metrics.py -r /home/zzf/codebase/repo_VoxSRC20-evaltoolkits/voxconverse/*.rttm -s ./*.rttm