python train_classical.py --model svm --dataset-name sel-10-undersampled
python train_classical.py --model svm --dataset-name all-undersampled
python train_classical.py --model svm --dataset-name sel-10
python train_classical.py --model svm --dataset-name all

python train_classical.py --model dt --dataset-name sel-10-undersampled
python train_classical.py --model dt --dataset-name all-undersampled
python train_classical.py --model dt --dataset-name sel-10
python train_classical.py --model dt --dataset-name all

python train_classical.py --model rf --dataset-name sel-10-undersampled
python train_classical.py --model rf --dataset-name all-undersampled
python train_classical.py --model rf --dataset-name sel-10
python train_classical.py --model rf --dataset-name all

python train_classical.py --model gb --dataset-name sel-10-undersampled
python train_classical.py --model gb --dataset-name all-undersampled
python train_classical.py --model gb --dataset-name sel-10
python train_classical.py --model gb --dataset-name all

shutdown.exe -s -t 0
