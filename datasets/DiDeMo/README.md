---
configs:
- config_name: default
  data_files:
  - split: train
    path: "didemo_train.json"
  - split: test
    path: "didemo_test.json"
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

[DiDeMo](https://openaccess.thecvf.com/content_iccv_2017/html/Hendricks_Localizing_Moments_in_ICCV_2017_paper.html) contains 10K long-form videos from Flickr. For each video, ~4 short sentences are annotated in temporal order. We follow the existing works to concatenate those short sentences and evaluate ‘paragraph-to-video’ retrieval on this benchmark.

We adopt the official split:  
- Train:  8,395 videos, 8,395 captions (concatenate from 33,005 short captions)  
- Val: 1,065 videos, 1,065 captions (concatenate from 4,290 short captions) (We don't have the collection yet.)  
- Test: 1,004 videos, 1,004 captions (concatenate from 4,021 short captions)  

---

## Get Raw Videos

```bash
cat DiDeMo_Videos_mp4_train.tar.part-* | tar -vxf -
tar -xvf DiDeMo_Videos_mp4_test.tar
```

---

## Official Release

Video Release: [DiDeMoRelease](https://data.ciirc.cvut.cz/public/projects/LisaAnne/DiDeMoRelease/)

---

## 🌟 Citation

```bibtex
@inproceedings{anne2017didemo,
  title={Localizing moments in video with natural language},
  author={Anne Hendricks, Lisa and Wang, Oliver and Shechtman, Eli and Sivic, Josef and Darrell, Trevor and Russell, Bryan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2017}
}
```