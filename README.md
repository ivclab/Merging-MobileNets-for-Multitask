# Merging-MobileNets-for-Multitask

Official implementation of [On Merging MobileNets for Efficient Multitask Inference](https://docs.wixstatic.com/ugd/42e7ad_1e56b18cd2f04c198550ceafee3b7685.pdf).

Created by [Cheng-En Wu](https://github.com/CEWu) , Yi-Ming Chan(yiming@iis.sinica.edu.tw), Chu-Song Chen(song@iis.sinica.edu.tw)

## Usage
Pretrained and merged checkpoints are available here: https://reurl.cc/96o3x

Datasets in TFRecords are available here: https://reurl.cc/vp3vN   
  All rights belong to the respective publishers. The datasets are provided only to aid reproducibility.
  

Place and unzipped pretrained checkpoints `checkpoints.tar.gz` in checkpoints/ , 
      `datasets.tar.gz` in datasets/ `hungarian_algorithm.tar.gz` in hungarian_algorithm/

### Mergeing
Check out `convert_ckpt_to_npy.sh` and `merge_layers.sh`

Convert pretrained checkpoints to numpy files and merge pretrained checkpoints into a unified checkpoints. Training
Check out `zipper_multiple_train.sh`

Excute training in zippering process.
### Inference
Check out `zipper_eval_script.sh`

Evaluate the Top-1 accuracy.

## Citation
Please cite following paper if these codes help your research:

    @inproceedings{
      Title   = {On Merging MobileNets for Efficient Multitask Inference},
      Author  = {Cheng-En Wu, Yi-Ming Chan and Chu-Song Chen}, 
      booktitle = {2019 IEEE International Symposium on High-Performance Computer Architecture Workshop},
      year    = {2019}
    }
     
    
## Contact
Please feel free to leave suggestions or comments to [Cheng-En Wu](https://github.com/CEWu) , Yi-Ming Chan(yiming@iis.sinica.edu.tw), Chu-Song Chen(song@iis.sinica.edu.tw)
