# Selective Prediction on VQA


 **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
**Generate Raw logits and save them on disk**
```
Execute Selective_Prediction_VQA/predictions/run_prediction.ipynb
```

**Train Calibrator from raw logits**
```
Execute Selective_Prediction_VQA/calibration_methods/vector_scaling.ipynb
```

**Find &theta; at various desired risks for Vanilla MaxProb and Vector-scaling Calibrated logit MaxProb**
```
Execute Selective_Prediction_VQA/risk_bounds/compute_risk_bounds.ipynb
```

**Experiments**

1. **Risk Control Results with Maxprob for VQAv2 for &delta; = 0.001** 


| Desired Risk | Train Risk | Train Coverage | Test Risk | Test Coverage | Bounded Risk |
|--------------|------------|----------------|-----------|----------------|--------------|
| 0.02         | 0.0173     | 0.3624         | 0.0177    | 0.3649         | 0.0200       |
| 0.10         | 0.0957     | 0.6810         | 0.0980    | 0.6810         | 0.1000       |
| 0.15         | 0.1453     | 0.8013         | 0.1420    | 0.8002         | 0.1500       |
| 0.20         | 0.1950     | 0.9133         | 0.1940    | 0.9125         | 0.2000       |
| 0.25         | 0.2448     | 0.9996         | 0.2432    | 0.9996         | 0.2500       |
| 0.30         | 0.2444     | 1.0000         | 0.2442    | 1.0000         | 0.2496       |
| 1.00         | 0.2440     | 1.0000         | 0.2446    | 1.0000         | 0.2491       |


2. **Risk Control Results with Vector-Scaling Calibration for VQAv2 for &delta; = 0.001** 

| Desired Risk | Train Risk | Train Coverage | Test Risk | Test Coverage | Bounded Risk |
|--------------|------------|----------------|-----------|---------------|--------------|
| 0.02         | 0.0000     | 0.0000         | 0.0303    | 0.4585        | 0.9999       |
| 0.10         | 0.0957     | 0.6785         | 0.0961    | 0.6796        | 0.1000       |
| 0.15         | 0.1452     | 0.7909         | 0.1436    | 0.7905        | 0.1500       |
| 0.20         | 0.1950     | 0.8976         | 0.1937    | 0.8975        | 0.2000       |
| 0.25         | 0.2448     | 0.9996         | 0.2445    | 0.9997        | 0.2500       |
| 0.30         | 0.2446     | 1.0000         | 0.2453    | 1.0000        | 0.2497       |
| 1.00         | 0.2461     | 1.0000         | 0.2437    | 1.0000        | 0.2513       |

**For more details, please read the [report](report/MAI_Project_Report_GroupE.pdf) :smiley:**

## References:

+ https://github.com/geifmany/selective_deep_learning \
+ https://huggingface.co/datasets/HuggingFaceM4/VQAv2/blob/main/VQAv2.py \
+ https://github.com/saurabhgarg1996/calibration

### Citations:
  ```bibtex
  @ARTICLE{2017arXiv170508500G,
        author = {{Geifman}, Y. and {El-Yaniv}, R.},
        title = "{Selective Classification for Deep Neural Networks}",
        journal = {ArXiv e-prints},
        archivePrefix = "arXiv",
        eprint = {1705.08500},
        year = 2017
    }

  
  @misc{kim2021vilt,
        title={ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision}, 
        author={Wonjae Kim and Bokyung Son and Ildoo Kim},
        year={2021},
        eprint={2102.03334},
        archivePrefix={arXiv},
        primaryClass={stat.ML}
  }

  
