# Lidar-Camera Semi-Supervised Learning for Semantic Segmentation - Data splits 

Public release of our data splits used for the results in the paper: 
```bibtex
@ARTICLE{caltagirone2021dataset,
    author = {Caltagirone, Luca and Bellone, Mauro and Lennart, Svenson and Mattias, Wahde and Raivo, Sell},
     title = {Lidar-Camera Semi-Supervised Learning for Semantic Segmentation},
   journal = {MDPI sensors},
    volume = {0},
    number = {0},
      year = {2021},
       url = {https://},
       doi = {----add----},
     pages = {000--000}
}
```



## Dataset splits
In our paper, we used the [Waymo dataset] which consists of 1110 driving sequences collected under various weather and lighting conditions. 
The Waymo open dataset includes 1110 driving sequences recorded with multiple cameras and lidars across a large variety of locations, road types and weather and lighting conditions. 
Each driving sequence consists of a 20-s-long recording sampled at 10 Hz. Both 2D and 3D bounding boxes were manually generated for all frames and considering the following four categories of objects: vehicles; pedestrians; cyclists; and traffic signs.
Additionally, we have partitioned the driving sequences into four broad subsets, namely day–fair; night–fair; day–rain; night–rain (see Table below for further details). 

| Day–Fair | Day–Rain| Night–Fair | Night–Rain |
| ------ | ------ |------ |------ |
| 747 | 226 | 82 | 45 |
> table: Number of 20-s-long driving sequences belonging to the four main categories considered in
this work (10 Hz sample frequency).


The labels day and night indicate whether a sequence was collected during the day under good lighting conditions, or late in the day or at night under poor external illumination. 
The labels fair and rain instead refer to the weather conditions, with fair denoting good weather, and rain denoting active raining or wet environment following recent precipitation.
Each sequence originally contains approximately 200 frames (i.e., 20 s sampled at 10 Hz) of which we kept every 10th frame. 
Out of the full dataset, for~the experiments described in the following sections, we generated N=10 random dataset splits.
The randomization was carried out by considering driving sequences instead of individual frames in order to avoid any overlap between the subsets in any given set. 

With the exception of the training sets, which only contain examples belonging to the coarse category day--fair, all other sets include examples belonging to all coarse categories. 
By only considering training data belonging to one category, it is possible to investigate the effectiveness of the proposed approach for carrying out domain adaptation. 
(See Table below for more details regarding the dataset splits S.)


|Set | Day–Fair | Day–Rain| Night–Fair | Night–Rain |
|------ | ------ | ------ |------ |------ |
|Training| 100 | 0 | 0 | 0 |
|Validation| 10 | 10 | 10 | 5 |
|Unlabelled| 40 | 40 | 36 | 20 |
|Test| 40 | 40 | 36 | 20 |
> table: Number of sequences assigned to training, validation, unlabelled, and test subsets for each
dataset split Si according to the four main categories considered in this work.


The specific lists are provided in the folder ./data/splits/. as text files

```bash
./data/splits/
```

The list files contain rows in the following format: 

```bash
labeled/day/not_rain/camera/segment-575209926587730008_3880_000_3900_000_with_camera_labels_0000000124.png
```

where "day", "not_rain" etc. refer to the specific split in the table above, while "segment-XXXXX-.png" refers to the specific image from the waymo dataset. File names follow the same standard and the same format. 
