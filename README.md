## [Official Code] LEFT: Efficient Large Entries Retrieval
This is the code for the paper "LEFT: Efficient Large Entries Retrieval in Network Monitoring".

## Abstract
balablablabalbal

## Requirements
PyTorch, Loguru, Faiss, Lightning

## Experiments

You can try to reproduce the main results in this paper by running the following commands.

1. For model selection (Table I)
    ```
    sh run_math.sh
    sh run_ntc.sh
    ```

2. For retrieval performance comparison and run time (Table II & IV)
    ```
    sh run_ann.sh
    sh run_left.sh
    ```

3. For hyper-parameters (Table III)
    ```
    sh run_hyper.sh
    ```


## License

Detectron2 is released under the [Apache 2.0 license](LICENSE).

## Citing LEFT
If you use LEFT in your research or wish to refer to the content, please use the following BibTeX entry. 

```BibTeX
@Article{LightNestle,
  title   = {LightNestle: Quick and Accurate Neural Sequential Tensor Completion via Meta Learning},
  author = {Li, Yuhui and Liang, Wei and Xie, Kun and Zhang, Dafang and Xie, Songyou and Li, Kuan-Ching},
  booktitle = {{{IEEE INFOCOM}} 2023 - {{IEEE Conference}} on {{Computer Communications}}},
  year    = {2023},
}
```
