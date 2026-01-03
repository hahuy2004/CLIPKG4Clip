---
configs:
- config_name: default
  data_files:
  - split: train
    path: "activitynet_captions_train.json"
  - split: val1
    path: "activitynet_captions_val1.json"
  - split: val2
    path: "activitynet_captions_val2.json"
task_categories:
- text-to-video
- text-retrieval
- video-classification
language:
- en
size_categories:
- 10K<n<100K
---

## About

[ActivityNet Captions](https://openaccess.thecvf.com/content_iccv_2017/html/Krishna_Dense-Captioning_Events_in_ICCV_2017_paper.html) contains 20K long-form videos (180s as average length) from YouTube and 100K captions. Most of the videos contain over 3 annotated events. We follow the existing works to concatenate multiple short temporal descriptions into long sentences and evaluate ‘paragraph-to-video’ retrieval on this benchmark.

We adopt the official split:  
- **Train:**  10,009 videos, 10,009 captions (concatenate from 37,421 short captions)  
- **Test (Val1):** 4,917 videos, 4,917 captions (concatenate from 17,505 short captions)  
- **Val2:** 4,885 videos, 4,885 captions (concatenate from 17,031 short captions)  

---

## Get Raw Videos

```bash
cat ActivityNet_Videos.tar.part-* | tar -vxf -
```

---

## Official Release

ActivityNet Official Release: [ActivityNet Download](http://activity-net.org/download.html)

---

## 🌟 Citation

```bibtex
@inproceedings{caba2015activitynet,
  title={Activitynet: A large-scale video benchmark for human activity understanding},
  author={Caba Heilbron, Fabian and Escorcia, Victor and Ghanem, Bernard and Carlos Niebles, Juan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2015}
}
```