# utp_alignment - Python code for utp visual alignment
## Getting Started
Install packages:
```bash  
pip install -r requirements.txt
```
Download datasets including images of chips and detectors from [dataset](https://pan.baidu.com/s/1sjh231tbPVtWMzXxWxX85Q?pwd=wv17),
unzip the package and put dataset folders into "./".<br>
Create "result" folder in "./"
## Train
To train the model, run:
```bash  
python train_multi.py
```
Results will be saved in ./results/train_results, params will be saved in ./results/best_params.json
## Test
To test the model, run:
```bash  
python test_multi.py
```
Results will be saved in ./results/test_results<br>
**Current results**
![gc](current_results_visulization/test_0000_2025-12-25-21-03-06.jpg.png 'gc')
![detector](current_results_visulization/result_detector_2023-12-06-11-10-47.png 'detector')