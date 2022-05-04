clean:
	rm -rf ./checkpoint

download:
	rm -f checkpoint/*.cp
	mkdir -p checkpoint
	scp -r vagrant@192.168.121.25:/vagrant/data/FunPos/checkpoint/* checkpoint/

upload:
	ssh vagrant@192.168.121.25 'rm -rf /vagrant/data/FunPos'
	scp -r "$$PWD" vagrant@192.168.121.25:/vagrant/data
	ssh vagrant@192.168.121.25 'rm -rf /vagrant/data/FunPos/checkpoint'

train:
	python3 train.py

test:
	python3 test.py

remote_train: upload
	ssh vagrant@192.168.121.25 'cd /vagrant/data/FunPos && python3 train.py'

labels:
	python3 label.py
