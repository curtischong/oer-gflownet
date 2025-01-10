train:
	python main.py env=oerenv proxy=oerproxy
train_overpotential:
	python gflownet/proxy/train_overpotential_predictor.py