# EF (Encoded Fusion) Weights Directory

This directory stores the EF (Encoded Fusion) weights for the different segmentation models and encoding dimensions.


- **Segformer_21**: 21 classes (for ScanNet dataset)
- **Segformer_101**: 101 classes (for ScanNet++ dataset) 
- **Segformer_150**: 150 classes (for BS3D dataset)

## Directory Structure

```
EF_weights/
|-- ESANet_21/
│   |-- dim_2/     # Place ESANet 21-class dim=2 EF weights here
│   |-- dim_4/     # Place ESANet 21-class dim=4 EF weights here
│   |-- dim_8/     # Place ESANet 21-class dim=8 EF weights here
|-- segformer_21/          # 21 classes (ScanNet)
│   |-- dim_2/     # Place Segformer 21-class dim=2 EF weights here
│   |-- dim_4/     # Place Segformer 21-class dim=4 EF weights here
│   |-- dim_8/     # Place Segformer 21-class dim=8 EF weights here
|-- segformer_101/         # 101 classes (ScanNet++)
│   |-- dim_2/     # Place Segformer 101-class dim=2 EF weights here
│   |-- dim_4/     # Place Segformer 101-class dim=4 EF weights here
│   |-- dim_8/     # Place Segformer 101-class dim=8 EF weights here
|-- segformer_150/         # 150 classes (BS3D)
    |-- dim_2/     # Place Segformer 150-class dim=2 EF weights here
    |-- dim_4/     # Place Segformer 150-class dim=4 EF weights here
    |-- dim_8/     # Place Segformer 150-class dim=8 EF weights here
```
