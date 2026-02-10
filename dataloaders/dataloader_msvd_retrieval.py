from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pickle

import random
import torch
from PIL import Image
from os.path import exists, join
from collections import defaultdict, OrderedDict
# Try to import Decord for TempMe mode
try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    print("[Warning] Decord not available. TempMe mode will be disabled. Install: pip install decord")

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode, RandomHorizontalFlip, RandomResizedCrop
from dataloaders.rawvideo_util import RawVideoExtractor

# Try to import TempMe-specific modules
try:
    import dataloaders.video_transforms as video_transforms
    from .random_erasing import RandomErasing
    TEMPME_MODULES_AVAILABLE = True
except ImportError:
    TEMPME_MODULES_AVAILABLE = False
    print("[Warning] TempMe modules (video_transforms, random_erasing) not available.")

class MSVD_DataLoader(Dataset):
    """MSVD dataset loader."""
    def __init__(
            self,
            subset,
            data_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            use_enriched=False,
    ):
        self.data_path = data_path
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.subset = subset
        assert self.subset in ["train", "val", "test"]
        video_id_path_dict = {}
        video_id_path_dict["train"] = os.path.join(self.data_path, "train_list.txt")
        video_id_path_dict["val"] = os.path.join(self.data_path, "val_list.txt")
        video_id_path_dict["test"] = os.path.join(self.data_path, "test_list.txt")
        
        # Choose caption file based on whether using enriched data
        if use_enriched and self.subset == "train":
            enriched_caption_file = os.path.join(self.data_path, "enriched-caption-complete.pkl")
            if os.path.exists(enriched_caption_file):
                caption_file = enriched_caption_file
                print(f"Using ENRICHED captions: {caption_file}")
            else:
                caption_file = os.path.join(self.data_path, "raw-captions.pkl")
                print(f"Enriched caption file not found, using raw captions: {caption_file}")
        else:
            caption_file = os.path.join(self.data_path, "raw-captions.pkl")
            print(f"Using RAW captions: {caption_file}")

        with open(video_id_path_dict[self.subset], 'r') as fp:
            video_ids = [itm.strip() for itm in fp.readlines()]

        with open(caption_file, 'rb') as f:
            captions = pickle.load(f)

        video_dict = {}
        for root, dub_dir, video_files in os.walk(self.features_path):
            for video_file in video_files:
                video_id_ = ".".join(video_file.split(".")[:-1])
                if video_id_ not in video_ids:
                    continue
                file_path_ = os.path.join(root, video_file)
                video_dict[video_id_] = file_path_
        self.video_dict = video_dict

        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = []
        for video_id in video_ids:
            assert video_id in captions
            for cap in captions[video_id]:
                cap_txt = " ".join(cap)
                self.sentences_dict[len(self.sentences_dict)] = (video_id, cap_txt)
            self.cut_off_points.append(len(self.sentences_dict))

        ## below variables are used to multi-sentences retrieval
        # self.cut_off_points: used to tag the label when calculate the metric
        # self.sentence_num: used to cut the sentence representation
        # self.video_num: used to cut the video representation
        self.multi_sentence_per_video = True    # !!! important tag for eval
        if self.subset == "val" or self.subset == "test":
            self.sentence_num = len(self.sentences_dict)
            self.video_num = len(video_ids)
            assert len(self.cut_off_points) == self.video_num
            print("For {}, sentence number: {}".format(self.subset, self.sentence_num))
            print("For {}, video number: {}".format(self.subset, self.video_num))

        print("Video number: {}".format(len(self.video_dict)))
        print("Total Paire: {}".format(len(self.sentences_dict)))

        self.sample_len = len(self.sentences_dict)
        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(caption)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=float)

        for i, video_id in enumerate(choice_video_ids):
            video_path = self.video_dict[video_id]

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']

            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        video_id, caption = self.sentences_dict[idx]

        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
        video, video_mask = self._get_rawvideo(choice_video_ids)
        return pairs_text, pairs_mask, pairs_segment, video, video_mask

# ============================================================================
# TempMe-based Classes (Advanced Data Augmentation + Decord)
# ============================================================================

class RetrievalDataset_TempMe(Dataset):
    """TempMe-style General Retrieval Dataset with Decord and advanced augmentation."""

    def __init__(
            self,
            subset,
            anno_path,
            video_path,
            tokenizer,
            max_words=30,
            max_frames=12,
            video_framerate=1,
            image_resolution=224,
            mode='all',
            config=None
    ):
        if not DECORD_AVAILABLE:
            raise ImportError(
                "Decord is required for TempMe mode. Install: pip install decord"
            )
        if not TEMPME_MODULES_AVAILABLE:
            raise ImportError(
                "TempMe modules (video_transforms, random_erasing) are required."
            )
        
        self.subset = subset
        self.anno_path = anno_path
        self.video_path = video_path
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.max_frames = max_frames
        self.video_framerate = video_framerate
        self.image_resolution = image_resolution
        self.mode = mode  # all/text/vision
        self.config = config

        self.video_dict, self.sentences_dict = self._get_anns(self.subset)

        self.video_list = list(self.video_dict.keys())
        self.sample_len = 0

        print("Video number: {}".format(len(self.video_dict)))
        print("Total Pairs: {}".format(len(self.sentences_dict)))

        # Use TempMe-style extractor with augmentation
        self.rawVideoExtractor = RawVideoExtractor(use_tempme=True, framerate=video_framerate, 
                                                   size=image_resolution, subset=subset)
        self.transform = Compose([
            Resize(image_resolution, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_resolution),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.tsfm_dict = {
            'clip_test': Compose([
                Resize(image_resolution, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_resolution),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]),
            'clip_train': Compose([
                RandomResizedCrop(image_resolution, scale=(0.5, 1.0)),
                RandomHorizontalFlip(),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        }
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.image_resolution = image_resolution
        if self.mode in ['all', 'text']:
            self.sample_len = len(self.sentences_dict)
        else:
            self.sample_len = len(self.video_list)
        self.aug_transform = video_transforms.create_random_augment(
            input_size=(self.image_resolution, self.image_resolution),
            auto_augment='rand-m7-n4-mstd0.5-inc1',
            interpolation='bicubic',
        )

    def __len__(self):
        return self.sample_len
    
    def __aug_transform(self, buffer):
        _aug_transform = video_transforms.create_random_augment(
            input_size=(self.image_resolution, self.image_resolution),
            auto_augment='rand-m7-n4-mstd0.5-inc1',
            interpolation='bicubic',
        )
        buffer = _aug_transform(buffer)
        return buffer

    def _get_anns(self, subset='train'):
        raise NotImplementedError

    def _get_text(self, caption):
        if len(caption) == 3:
            _caption_text, s, e = caption
        else:
            raise NotImplementedError

        if isinstance(_caption_text, list):
            caption_text = random.choice(_caption_text)
        else:
            caption_text = _caption_text

        words = self.tokenizer.tokenize(caption_text)

        if self.subset == "train" and 0:
            if random.random() < 0.5:
                new_words = []
                for idx in range(len(words)):
                    if random.random() < 0.8:
                        new_words.append(words[idx])
                words = new_words

        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
        assert len(input_ids) == self.max_words
        assert len(input_mask) == self.max_words

        input_ids = np.array(input_ids)
        input_mask = np.array(input_mask)

        return input_ids, input_mask, s, e

    def _get_rawvideo(self, video_id, s=None, e=None):
        """Legacy method using OpenCV backend - kept for compatibility."""
        video_mask = np.zeros(self.max_frames, dtype=np.long)
        max_video_length = 0

        # T x 3 x H x W
        video = np.zeros((self.max_frames, 3, self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=float)

        if s is None:
            start_time, end_time = None, None
        else:
            start_time = int(s)
            end_time = int(e)
            start_time = start_time if start_time >= 0. else 0.
            end_time = end_time if end_time >= 0. else 0.
            if start_time > end_time:
                start_time, end_time = end_time, start_time
            elif start_time == end_time:
                end_time = end_time + 1
        video_path = self.video_dict[video_id]

        raw_video_data = self.rawVideoExtractor.get_video_data(video_path, start_time, end_time)
        raw_video_data = raw_video_data['video']

        if len(raw_video_data.shape) > 3:
            # L x T x 3 x H x W

            if self.max_frames < raw_video_data.shape[0]:
                sample_indx = np.linspace(0, raw_video_data.shape[0] - 1, num=self.max_frames, dtype=int)
                video_slice = raw_video_data[sample_indx, ...]
            else:
                video_slice = raw_video_data

            video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=0)

            slice_len = video_slice.shape[0]
            max_video_length = max_video_length if max_video_length > slice_len else slice_len
            if slice_len < 1:
                pass
            else:
                video[:slice_len, ...] = video_slice
        else:
            print("video path: {} error. video id: {}".format(video_path, video_id))

        video_mask[:max_video_length] = [1] * max_video_length

        return video, video_mask

    def _get_rawvideo_dec(self, video_id, s=None, e=None):
        """Decord-based video loading with advanced augmentation."""
        video_mask = np.zeros(self.max_frames, dtype=np.long)
        max_video_length = 0

        # T x 3 x H x W
        video = np.zeros((self.max_frames, 3, self.image_resolution, self.image_resolution), dtype=float)

        if s is None:
            start_time, end_time = None, None
        else:
            start_time = int(s)
            end_time = int(e)
            start_time = start_time if start_time >= 0. else 0.
            end_time = end_time if end_time >= 0. else 0.
            if start_time > end_time:
                start_time, end_time = end_time, start_time
            elif start_time == end_time:
                end_time = start_time + 1
        video_path = self.video_dict[video_id]

        if exists(video_path):
            vreader = VideoReader(video_path, ctx=cpu(0))
        else:
            print(video_path)
            raise FileNotFoundError

        fps = vreader.get_avg_fps()
        f_start = 0 if start_time is None else int(start_time * fps)
        f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
        num_frames = f_end - f_start + 1
        if num_frames > 0:
            # T x 3 x H x W
            sample_fps = int(self.video_framerate)
            t_stride = int(round(float(fps) / sample_fps))

            all_pos = list(range(f_start, f_end + 1, t_stride))
            if len(all_pos) > self.max_frames:
                sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=self.max_frames, dtype=int)]
            elif len(all_pos) == self.max_frames:
                sample_pos = all_pos
            else:
                sample_pos = list(np.linspace(f_start, f_end, num=self.max_frames, dtype=int))
            assert len(sample_pos) == self.max_frames
            
            patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
            if self.subset == "train":
                # for i in range(2):
                patch_images = self.aug_transform(patch_images)

            # if self.subset == "train":
            #     patch_images = torch.stack([self.tsfm_dict["clip_train"](img) for img in patch_images])
            # else:
            #     patch_images = torch.stack([self.tsfm_dict["clip_test"](img) for img in patch_images])

            patch_images = torch.stack([self.transform(img) for img in patch_images])
            slice_len = patch_images.shape[0]
            max_video_length = max_video_length if max_video_length > slice_len else slice_len
            if slice_len < 1:
                pass
            else:
                video[:slice_len, ...] = patch_images
        else:
            print("video path: {} error. video id: {}".format(video_path, video_id))

        video_mask[:max_video_length] = [1] * max_video_length

        return video, video_mask

    def __getitem__(self, idx):
        if self.mode == 'all':
            video_id, caption = self.sentences_dict[idx]
            text_ids, text_mask, s, e = self._get_text(caption)
            video, video_mask = self._get_rawvideo_dec(video_id, s, e)
            # video, video_mask = self._get_rawvideo(video_id, s, e)
            return text_ids, text_mask, video, video_mask, idx, hash(video_id.replace("video", ""))
        elif self.mode == 'text':
            video_id, caption = self.sentences_dict[idx]
            text_ids, text_mask, s, e = self._get_text(caption)
            return text_ids, text_mask, idx
        elif self.mode == 'video':
            video_id = self.video_list[idx]
            video, video_mask = self._get_rawvideo_dec(video_id)
            # video, video_mask = self._get_rawvideo(video_id)
            return video, video_mask, idx

    def get_text_len(self):
        return len(self.sentences_dict)

    def get_video_len(self):
        return len(self.video_list)

    def get_text_content(self, ind):
        return self.sentences_dict[ind][1]

    def get_data_name(self):
        return self.__class__.__name__ + "_" + self.subset

    def get_vis_info(self, idx):
        video_id, caption = self.sentences_dict[idx]
        video_path = self.video_dict[video_id]
        return caption, video_path


class MSVDDataset_TempMe(RetrievalDataset_TempMe):
    """TempMe-style MSVD dataset with Decord and advanced augmentation."""

    def __init__(self, subset, anno_path, video_path, tokenizer, max_words=32,
                 max_frames=12, video_framerate=1, image_resolution=224, mode='all', config=None,
                 use_enriched=False):
        """
        Initialize MSVD TempMe dataset.
        
        Args:
            subset: 'train', 'val', or 'test'
            anno_path: Path to directory containing annotation files (train_list.txt, captions.pkl, etc.)
            video_path: Path to directory containing video features
            tokenizer: Text tokenizer
            max_words: Maximum number of words in text
            max_frames: Maximum number of frames to sample
            video_framerate: Video sampling frame rate
            image_resolution: Image resolution
            mode: 'all', 'text', or 'video'
            config: Additional config
            use_enriched: Whether to use enriched captions (default: False)
        """
        self.use_enriched = use_enriched
        super(MSVDDataset_TempMe, self).__init__(subset, anno_path, video_path, tokenizer, max_words,
                                                  max_frames, video_framerate, image_resolution, mode, config=config)

    def _get_anns(self, subset='train'):
        """
        Load MSVD annotations.
        
        Returns:
            video_dict: OrderedDict mapping video_id -> video_path
            sentences_dict: OrderedDict mapping idx -> (video_id, [caption_text, start, end])
        
        Note: MSVD doesn't have timestamps, so start=None, end=None for full video.
        """
        # Define paths to video list files
        video_id_path_dict = {
            'train': join(self.anno_path, 'train_list.txt'),
            'val': join(self.anno_path, 'val_list.txt'),
            'test': join(self.anno_path, 'test_list.txt')
        }
        
        # Read video IDs for this subset
        video_list_path = video_id_path_dict[subset]
        if not exists(video_list_path):
            raise FileNotFoundError(f"Video list file not found: {video_list_path}")
        
        with open(video_list_path, 'r') as fp:
            video_ids = [line.strip() for line in fp.readlines()]
        
        # Choose caption file based on whether using enriched data
        if self.use_enriched and subset == "train":
            enriched_caption_file = join(self.anno_path, "enriched-caption-complete.pkl")
            if exists(enriched_caption_file):
                caption_file = enriched_caption_file
                print(f"[MSVDDataset_TempMe] Using ENRICHED captions: {caption_file}")
            else:
                caption_file = join(self.anno_path, "raw-captions.pkl")
                print(f"[MSVDDataset_TempMe] Enriched caption file not found, using raw captions: {caption_file}")
        else:
            caption_file = join(self.anno_path, "raw-captions.pkl")
            print(f"[MSVDDataset_TempMe] Using RAW captions: {caption_file}")
        
        # Load captions from pickle file
        if not exists(caption_file):
            raise FileNotFoundError(f"Caption file not found: {caption_file}")
        
        with open(caption_file, 'rb') as f:
            captions = pickle.load(f)
        
        # Build video_dict: video_id -> video_path
        video_dict = OrderedDict()
        for root, dub_dir, video_files in os.walk(self.video_path):
            for video_file in video_files:
                video_id_ = ".".join(video_file.split(".")[:-1])
                if video_id_ not in video_ids:
                    continue
                file_path_ = join(root, video_file)
                video_dict[video_id_] = file_path_
        
        # Build sentences_dict: idx -> (video_id, [caption_text, start, end])
        # MSVD doesn't have timestamps, so we use start=0, end=0
        sentences_dict = OrderedDict()
        for video_id in video_ids:
            if video_id not in captions:
                print(f"[Warning] Video {video_id} has no captions, skipping...")
                continue
            
            # Each video can have multiple captions
            for cap in captions[video_id]:
                # cap is a list of words, join them to create caption text
                if isinstance(cap, list):
                    cap_txt = " ".join(cap)
                else:
                    cap_txt = cap
                
                # Format: (video_id, [caption_text, start, end])
                # MSVD has no timestamps, so use None to indicate full video
                sentences_dict[len(sentences_dict)] = (video_id, [cap_txt, None, None])
        
        print(f"[MSVDDataset_TempMe/{subset}] Loaded {len(video_dict)} videos, {len(sentences_dict)} text-video pairs")
        
        # Check for unique captions
        unique_captions = set([v[1][0] for v in sentences_dict.values()])
        print(f'[MSVDDataset_TempMe/{subset}] Unique captions: {len(unique_captions)}, Total pairs: {len(sentences_dict)}')
        
        # ============================================================================
        # CRITICAL: Multi-sentence evaluation logic (required for MSVD eval)
        # ============================================================================
        # MSVD has multiple captions per video (~41 captions/video)
        # eval_epoch() needs these attributes to correctly reshape similarity matrix
        self.multi_sentence_per_video = True
        
        if subset == "val" or subset == "test":
            # Calculate cut_off_points: cumulative caption count per video
            # Used by eval_epoch() to group captions by video
            self.cut_off_points = []
            current_count = 0
            
            # CRITICAL FIX: Must iterate in SAME ORDER and SAME LOGIC as sentences_dict building
            # Only count videos that actually have captions (same as sentences_dict)
            for video_id in video_ids:
                if video_id not in captions:
                    continue  # Skip videos without captions (same as sentences_dict)
                
                n_caps = len(captions[video_id])
                current_count += n_caps
                self.cut_off_points.append(current_count)
            
            self.sentence_num = len(sentences_dict)
            # CRITICAL: video_num MUST equal number of videos WITH captions
            self.video_num = len(self.cut_off_points)
            
            # Validation check
            assert len(self.cut_off_points) == self.video_num, \
                f"cut_off_points length ({len(self.cut_off_points)}) != video_num ({self.video_num})"
            
            print(f"[MSVDDataset_TempMe/{subset}] Multi-sentence mode: {self.sentence_num} sentences, {self.video_num} videos")
            print(f"[MSVDDataset_TempMe/{subset}] Cut-off points (first 5): {self.cut_off_points[:5]}")
        # ============================================================================
        
        return video_dict, sentences_dict


# ============================================================================
# Factory Functions
# ============================================================================

def MSVDDataLoader(use_tempme=False, **kwargs):
    """
    Factory function to create the appropriate MSVD dataloader.
    
    Args:
        use_tempme (bool): If True, use TempMe-style dataloader with Decord and augmentation.
                          If False, use original CLIPKG4Clip dataloader.
        **kwargs: Arguments passed to the dataloader constructor.
        
    Returns:
        MSVD_DataLoader or MSVDDataset_TempMe instance.
        
    Example:
        # Original dataloader
        loader = MSVDDataLoader(use_tempme=False, subset="train", data_path="...", features_path="...", ...)
        
        # TempMe dataloader  
        loader = MSVDDataLoader(use_tempme=True, subset="train", anno_path="...", video_path="...", use_enriched=True, ...)
    """
    if use_tempme:
        return MSVDDataset_TempMe(**kwargs)
    else:
        return MSVD_DataLoader(**kwargs)

