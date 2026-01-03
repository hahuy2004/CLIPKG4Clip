---
configs:
- config_name: train_9k
  data_files:
  - split: train
    path: "msrvtt_train_9k.json"
- config_name: train_7k
  data_files:
  - split: train
    path: "msrvtt_train_7k.json"
- config_name: test_1k
  data_files:
  - split: test
    path: "msrvtt_test_1k.json"
task_categories:
- text-to-video
- text-retrieval
- video-classification
language:
- en
size_categories:
- 1K<n<10K
---


[MSRVTT](https://openaccess.thecvf.com/content_cvpr_2016/html/Xu_MSR-VTT_A_Large_CVPR_2016_paper.html) contains 10K video clips and 200K captions.

We adopt the standard `1K-A split` protocol, which was introduced in [JSFusion](https://openaccess.thecvf.com/content_ECCV_2018/html/Youngjae_Yu_A_Joint_Sequence_ECCV_2018_paper.html) and has since become the de facto benchmark split in the `Text-Video Retrieval` field.

Train:  
- train_7k: 7,010 videos, 140,200 captions  
- train_9k: 9,000 videos, 180,000 captions  

Test:  
- test_1k: 1,000 videos, 1,000 captions

---

## 🌟 Citation

```bibtex
@inproceedings{xu2016msrvtt,
  title={Msr-vtt: A large video description dataset for bridging video and language},
  author={Xu, Jun and Mei, Tao and Yao, Ting and Rui, Yong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2016}
}
```
