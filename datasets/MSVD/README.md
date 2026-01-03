---
configs:
- config_name: default
  data_files:
  - split: train
    path: "msvd_train.json"
  - split: validation
    path: "msvd_val.json"
  - split: test
    path: "msvd_test.json"
task_categories:
- text-to-video
- text-retrieval
- video-classification
language:
- en
size_categories:
- 1K<n<10K
---

[MSVD](https://aclanthology.org/P11-1020.pdf) contains 1,970 videos, each of which is paired with ~40 captions.

We adopt the official split:  
- Train:  1,200 videos, 48,774 captions  
- Val: 100 videos, 4,290 captions  
- Test: 670 videos, 27,763 captions

---

## 🌟 Citation

```bibtex
@inproceedings{chen2011collecting,
  title={Collecting highly parallel data for paraphrase evaluation},
  author={Chen, David and Dolan, William B},
  booktitle={Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2011}
}
```