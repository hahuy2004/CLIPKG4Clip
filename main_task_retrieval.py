from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import numpy as np
import random
import os
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
import time
import argparse
import datetime
import sys
from tqdm import tqdm
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4Clip, VTRModel, build_model
from modules.optimization import BertAdam

# TempMe: Import AdamW optimizer, AllGather and AllGather2
try:
    from modules.optimization_adamw import AdamW, get_cosine_schedule_with_warmup
    from modules.until_module import AllGather, AllGather2
    TEMPME_OPTIMIZER_AVAILABLE = True
except ImportError:
    TEMPME_OPTIMIZER_AVAILABLE = False
    AdamW = None
    get_cosine_schedule_with_warmup = None
    AllGather = None
    AllGather2 = None

from util import parallel_apply, get_logger
from dataloaders.data_dataloaders import DATALOADER_DICT

# TempMe: Import MetricLogger for advanced metrics tracking
try:
    from utils.metric_logger import MetricLogger
    METRICLOGGER_AVAILABLE = True
except ImportError:
    METRICLOGGER_AVAILABLE = False
    MetricLogger = None

# TempMe: Import communication utilities for distributed training
try:
    from utils.comm import is_main_process, synchronize
    TEMPME_COMM_AVAILABLE = True
except ImportError:
    TEMPME_COMM_AVAILABLE = False
    # Fallback implementations
    def is_main_process():
        return torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True
    def synchronize():
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

# Import enriched evaluation modules
from enriched_eval.fqs_selector import farthest_query_selection
from enriched_eval.aggregator import Aggregator

torch.distributed.init_process_group(backend="nccl")

global logger

def get_args(description='CLIPKG4Clip on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    
    # -----------------------------------------------------------------------------------------
    # Mode Selection: CLIPKG4Clip (original) vs TempMe (with LoRA & ToMe)
    # -----------------------------------------------------------------------------------------
    parser.add_argument('--use_tempme', action='store_true', 
                        help='Use TempMe mode (VTRModel with LoRA and ToMe). Default: False (CLIPKG4Clip mode)')
    
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument('--train_csv', type=str, default='data/.train.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')
    
    # TempMe-style data paths (alternative to csv paths)
    parser.add_argument('--anno_path', type=str, default=None, help='annotation path (TempMe mode)')
    parser.add_argument('--video_path', type=str, default=None, help='video path (TempMe mode)')
    
    # TempMe: Pretrained model path (required for TempMe mode)
    parser.add_argument('--pretrained_path', type=str, default='pretrained', 
                        help='Pretrained CLIP model path (TempMe mode only). Should contain ViT-B-32.pt, etc.')
    
    # Enriched data training parameters
    parser.add_argument('--enriched_data_path', type=str, default=None, help='Path to enriched captions JSON file for pre-training (MSRVTT)')
    parser.add_argument('--enriched', type=str, default='no', choices=['yes', 'no'], help='Use enriched captions for MSVD dataset')
    parser.add_argument('--enriched_epochs', type=int, default=3, help='Number of epochs to train on enriched data')
    parser.add_argument('--enriched_max_steps', type=int, default=-1, help='Max training steps for enriched data, -1 means no limit')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers (TempMe mode)')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--clip_lr', type=float, default=6e-4, help='learning rate for TempMe mode')
    parser.add_argument('--weight_decay', type=float, default=0.2, help='weight decay (used in TempMe mode)')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    # Arg --max_steps
    parser.add_argument('--max_steps', type=int, default=-1, help='max training steps, -1 means no limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    
    # TempMe compatibility: device and distributed args
    parser.add_argument("--device", default=None, type=str, help="Device: 'cuda' or 'cpu' (TempMe mode)")
    parser.add_argument("--distributed", default=0, type=int, help="Multi-machine DDP flag (TempMe mode)")
    
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    parser.add_argument('--loose_type', action='store_true', help="Default using tight type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="meanP",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help="choice a similarity header.")

    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")
    
    # TempMe compatibility: Add base_encoder as alias for pretrained_clip_name
    parser.add_argument("--base_encoder", default=None, type=str, 
                        help="CLIP backbone (TempMe mode). If not set, will use pretrained_clip_name. Example: 'ViT-B/32'")
    
    # TempMe compatibility: Add video_framerate as alias for feature_framerate
    parser.add_argument('--video_framerate', type=int, default=None, 
                        help='Framerate for video sampling (TempMe mode). If not set, will use feature_framerate.')

    # -----------------------------------------------------------------------------------------
    # TempMe (Token Merging) & LoRA Parameters (only used when --use_tempme is set)
    # -----------------------------------------------------------------------------------------
    parser.add_argument('--tome_r', type=int, default=0, 
                        help='Token reduction per layer (0=disabled, 2=recommended). Default: 0 (disabled for CLIPKG4Clip, 2 for TempMe)')
    parser.add_argument('--tome_tracesource', action='store_true', 
                        help='Trace source tokens in ToMe (for visualization/debugging). Default: False')
    parser.add_argument('--tome_propattn', action='store_true', default=True, 
                        help='Propagate attention weights in ToMe. Default: True (recommended)')
    
    parser.add_argument('--merge_layer', type=str, default='8-9-10', 
                        help='Layers to apply frame merging (e.g., "8-9-10"). TempMe default: "8-9-10"')
    parser.add_argument('--merge_frame_num', type=str, default='2-2-3', 
                        help='Number of frames to merge at each merge layer (e.g., "2-2-3"). TempMe default: "2-2-3"')
    parser.add_argument('--merge_token_proportion', type=str, default='30-10', 
                        help='Percentage of tokens to merge: frame-patch (e.g., "30-10" = 30%% frames, 10%% patches). TempMe default: "30-10"')
    
    parser.add_argument('--frame_pos', type=int, default=0, choices=[0, 1],
                        help='Enable frame position embedding (0=off, 1=on). TempMe default: 1. Default: 0 (disabled)')
    parser.add_argument('--lora_dim', type=int, default=0, 
                        help='LoRA dimension for visual encoder (0=disabled, 8=recommended). TempMe default: 8. Default: 0 (disabled)')
    
    # TempMe Data Augmentation
    parser.add_argument('--use_aug', action='store_true', 
                        help='Enable TempMe data augmentation (RandAugment + RandomErasing) for training. Default: False')
    # -----------------------------------------------------------------------------------------

    # Enriched evaluation parameters
    parser.add_argument('--eval_enriched', type=int, default=0, choices=[0, 1],
                        help='Use enriched queries: 0=no (baseline with val_csv), 1=yes (FQS with val_csv containing k+1 queries)')
    parser.add_argument('--aggregation_strategy', type=int, default=1, choices=[1, 2, 3],
                        help='Aggregation strategy: 1=Weighted RRF, 2=Average Similarity, 3=True Majority Voting (only when eval_enriched=1)')
    parser.add_argument('--fqs_k', type=int, default=2,
                        help='Number of enriched queries per video (default: 2, total k+1=3 queries)')

    args = parser.parse_args()

    if args.sim_header == "tightTransf":
        args.loose_type = False

    # Check parameters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    
    # TempMe mode validation
    if args.use_tempme:
        if not TEMPME_OPTIMIZER_AVAILABLE:
            raise ImportError(
                "TempMe mode requires optimization_adamw module. "
                "Please ensure modules/optimization_adamw.py is available."
            )
        if not os.path.exists(args.pretrained_path):
            raise FileNotFoundError(
                f"TempMe mode requires pretrained_path: {args.pretrained_path}. "
                "Please download CLIP pretrained models (e.g., ViT-B-32.pt)."
            )
        # Validate anno_path and video_path (required for TempMe dataloaders)
        if args.do_train and (args.anno_path is None or args.video_path is None):
            raise ValueError(
                "TempMe training requires --anno_path and --video_path. "
                "Example: --anno_path /data/MSRVTT/annotations --video_path /data/MSRVTT/videos"
            )
        # Set TempMe defaults if not specified (Note: logger not initialized yet, will log later)
        if args.tome_r == 0:
            args.tome_r = 2
        if args.lora_dim == 0:
            args.lora_dim = 8
        if args.frame_pos == 0:
            args.frame_pos = 1

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    # Validate enriched evaluation parameters
    if args.eval_enriched == 0 and hasattr(args, 'aggregation_strategy'):
        # Check if aggregation_strategy was explicitly set
        import sys
        if '--aggregation_strategy' in sys.argv:
            raise ValueError(
                "ERROR: --aggregation_strategy can only be used when --eval_enriched=1. "
                "You must enable enriched evaluation to use aggregation strategies."
            )

    return args

def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    rank = torch.distributed.get_rank()
    args.rank = rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def init_model(args, device, n_gpu, local_rank):
    """Initialize model based on mode (CLIPKG4Clip or TempMe)."""
    
    if args.use_tempme:
        # TempMe mode: Use VTRModel with LoRA and ToMe
        logger.info("Initializing VTRModel (TempMe mode)...")
        model = VTRModel(args)
        
        if args.init_model:
            if not os.path.exists(args.init_model):
                raise FileNotFoundError(f"Init model not found: {args.init_model}")
            logger.info(f"Loading TempMe checkpoint from {args.init_model}")
            model_state_dict = torch.load(args.init_model, map_location='cpu')
            model.load_state_dict(model_state_dict, strict=False)
        
        model.to(device)
    else:
        # CLIPKG4Clip mode: Use original CLIP4Clip
        logger.info("Initializing CLIP4Clip (original mode)...")
        if args.init_model:
            model_state_dict = torch.load(args.init_model, map_location='cpu')
        else:
            model_state_dict = None

        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)

    return model

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):
    """Prepare optimizer based on mode (CLIPKG4Clip or TempMe)."""
    
    if hasattr(model, 'module'):
        model = model.module

    if args.use_tempme:
        # TempMe mode: Only train LoRA parameters (TVPt_*)
        logger.info("Preparing TempMe optimizer (AdamW with LoRA parameters only)...")
        param_optimizer = list(model.named_parameters())
        
        # Only enable LoRA parameters
        for name, param in param_optimizer:
            if "TVPt" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        
        optimizer_parameters = []
        enabled_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                enabled_params.append(name)
                optimizer_parameters.append(param)
        
        logger.info(f"TempMe tuned parameters: {sorted(enabled_params)[:5]}... (total: {len(enabled_params)})")
        
        optimizer_grouped_params = [
            {'params': optimizer_parameters, 'lr': args.clip_lr}
        ]
        
        optimizer = AdamW(optimizer_grouped_params, weight_decay=args.weight_decay)
        num_warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_optimization_steps
        )
        
        find_unused = True  # TempMe has frozen parameters
    else:
        # CLIPKG4Clip mode: Original optimizer setup
        logger.info("Preparing CLIPKG4Clip optimizer (BertAdam)...")
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
        no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

        decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
        decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

        no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
        no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

        weight_decay = 0.2
        optimizer_grouped_parameters = [
            {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
            {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
            {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
            {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
        ]

        scheduler = None
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                             schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                             t_total=num_train_optimization_steps, weight_decay=weight_decay,
                             max_grad_norm=1.0)
        
        find_unused = False  # CLIPKG4Clip trains all parameters

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=find_unused)

    return optimizer, scheduler, model

def reduce_loss(loss, args):
    """Reduce loss across all processes for distributed training (TempMe mode)."""
    world_size = args.world_size if hasattr(args, 'world_size') else 1
    if world_size < 2:
        return loss
    with torch.no_grad():
        torch.distributed.reduce(loss, dst=0)
        if torch.distributed.get_rank() == 0:
            loss /= world_size
    return loss

def save_model(epoch, args, model, optimizer, tr_loss, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    optimizer_state_file = os.path.join(
        args.output_dir, "pytorch_opt.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': tr_loss,
            }, optimizer_state_file)
    logger.info("Model saved to %s", output_model_file)
    logger.info("Optimizer saved to %s", optimizer_state_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        model = None
    return model

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0, val_dataloader=None):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    early_stop = False
    
    # TempMe mode: Initialize MetricLogger for advanced tracking
    if args.use_tempme and METRICLOGGER_AVAILABLE:
        meters = MetricLogger(delimiter="  ")
        max_steps = len(train_dataloader) * args.epochs
        end = time.time()
    else:
        meters = None
        max_steps = None

    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        
        # Unpack batch - flexible for both CLIPKG4Clip and TempMe formats
        if args.use_tempme:
            # TempMe format: text_ids, text_mask, video, video_mask, inds, idx
            if len(batch) == 6:
                input_ids, input_mask, video, video_mask, inds, idx = batch
                segment_ids = torch.zeros_like(input_mask)  # TempMe doesn't use segment_ids
                loss = model(input_ids, input_mask, video, video_mask, idx, global_step + step)
            else:
                input_ids, input_mask, segment_ids, video, video_mask = batch
                loss = model(input_ids, segment_ids, input_mask, video, video_mask)
        else:
            # CLIPKG4Clip format: input_ids, input_mask, segment_ids, video, video_mask
            input_ids, input_mask, segment_ids, video, video_mask = batch
            loss = model(input_ids, segment_ids, input_mask, video, video_mask)

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            # https://github.com/openai/CLIP/issues/46
            if hasattr(model, 'module'):
                torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            else:
                torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

            global_step += 1
            
            # TempMe mode: Advanced logging with MetricLogger and ETA calculation
            if args.use_tempme and meters is not None:
                batch_time = time.time() - end
                data_time = 0  # Not tracking data loading time here
                reduced_loss = reduce_loss(loss.clone().detach(), args)
                meters.update(time=batch_time, data=data_time, loss=float(reduced_loss))
                
                # Get logit_scale for monitoring
                if hasattr(model, 'module'):
                    logit_scale = model.module.clip.logit_scale.exp().item()
                else:
                    logit_scale = model.clip.logit_scale.exp().item()
                
                if global_step % log_step == 0 and local_rank == 0:
                    eta_seconds = meters.time.global_avg * (max_steps - global_step)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    logger.info(
                        meters.delimiter.join([
                            "eta: {eta}",
                            "epoch: {epoch}/{max_epoch}",
                            "iteration: {iteration}/{max_iteration}",
                            "{meters}",
                            "lr: {lr}",
                            "logit_scale: {logit_scale:.2f}",
                            "max mem: {memory:.0f}",
                        ]).format(
                            eta=eta_string,
                            epoch=epoch + 1,
                            max_epoch=args.epochs,
                            iteration=global_step,
                            max_iteration=max_steps,
                            meters=str(meters),
                            lr="/".join([str('%.9f' % itm) for itm in sorted(list(set(scheduler.get_last_lr())))]),
                            logit_scale=logit_scale,
                            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        )
                    )
                
                # TempMe: Periodic evaluation during training + save best checkpoints
                if val_dataloader is not None and (global_step % (log_step * 3) == 0 or global_step == 1):
                    logger.info("Running evaluation at step {}...".format(global_step))
                    
                    # Use global variables for tracking
                    global best_score, best_score_list, sim_matrix_num, sim_name_list
                    
                    max_R1 = eval_epoch(args, model, val_dataloader, device, n_gpu)
                    
                    # max_R1 is a list for TempMe (supporting multiple similarity matrices)
                    if isinstance(max_R1, (list, tuple)):
                        if local_rank == 0:
                            for list_idx in range(min(len(max_R1), sim_matrix_num)):
                                if best_score_list[list_idx] < max_R1[list_idx]:
                                    best_score_list[list_idx] = max_R1[list_idx]
                                logger.info("The R1 is: {:.4f}\t| {:.4f}\tin {}".format(
                                    max_R1[list_idx], best_score_list[list_idx], 
                                    sim_name_list[list_idx] if list_idx < len(sim_name_list) else f"sim_{list_idx}"))
                            
                            # Save best checkpoint
                            if best_score < max(max_R1):
                                best_score = max(max_R1)
                                # TempMe-style save (no optimizer/tr_loss needed)
                                model_to_save = model.module if hasattr(model, 'module') else model
                                output_model_file = os.path.join(args.output_dir, "best.pth")
                                torch.save(model_to_save.state_dict(), output_model_file)
                                logger.info("Best model saved to %s", output_model_file)
                            
                            logger.info("The best R1 is: {:.4f} at all".format(best_score))
                    else:
                        # Single R1 score
                        if local_rank == 0 and max_R1 > best_score:
                            best_score = max_R1
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(args.output_dir, "best.pth")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Best model (R1={:.4f}) saved to {}".format(best_score, output_model_file))
                    
                    # Synchronize across GPUs
                    if TEMPME_COMM_AVAILABLE:
                        synchronize()
                    elif torch.distributed.is_initialized():
                        torch.distributed.barrier()
                    
                    model.train()  # Back to training mode
                    
                end = time.time()
            else:
                # CLIPKG4Clip mode: Original simple logging
                if global_step % log_step == 0 and local_rank == 0:
                    logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                                args.epochs, step + 1,
                                len(train_dataloader), "-".join([str('%.9f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                                float(loss),
                                (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                    start_time = time.time()
            
            # Kiểm tra nếu đã đạt max_steps
            if args.max_steps > 0 and global_step >= args.max_steps:
                if local_rank == 0:
                    logger.info("Reached max_steps: %d, stopping training.", args.max_steps)
                early_stop = True
                break

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step, early_stop

def _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list):
    sim_matrix = []
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        for idx2, b2 in enumerate(batch_list_v):
            video_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            b1b2_logits, *_tmp = model.get_similarity_logits(sequence_output, visual_output, input_mask, video_mask,
                                                                     loose_type=model.loose_type)
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    return sim_matrix

def eval_epoch(args, model, test_dataloader, device, n_gpu):
    """Evaluation function supporting both CLIPKG4Clip and TempMe modes."""
    
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    model.eval()
    
    # ========================================================================
    # TempMe Mode: Use stage1_eval + AllGather + stage2_eval pipeline
    # ========================================================================
    if args.use_tempme:
        logger.info("Running TempMe evaluation...")
        
        # Initialize allgather function
        if AllGather is None:
            raise RuntimeError("AllGather not available. Please check modules.until_module import.")
        allgather = AllGather.apply
        
        batch_cls, batch_mask_t = [], []
        batch_video_feat, batch_mask_v = [], []
        batch_ids = []
        
        with torch.no_grad():
            logger.info('[start] extract features')
            for batch in tqdm(test_dataloader, desc="Extracting features"):
                batch = tuple(t.to(device) for t in batch)
                
                # TempMe dataloader returns: text_ids, text_mask, video, video_mask, inds, idx
                if len(batch) == 6:
                    text_ids, text_mask, video, video_mask, inds, _ = batch
                else:
                    # Fallback: CLIPKG4Clip format
                    text_ids, text_mask, segment_ids, video, video_mask = batch
                    inds = torch.arange(text_ids.size(0)).to(device)
                
                # Stage 1: Extract features
                cls, video_feat = model.stage1_eval(text_ids, text_mask, video, video_mask)
                batch_cls.append(cls)
                batch_mask_t.append(text_mask)
                batch_video_feat.append(video_feat)
                batch_mask_v.append(video_mask)
                batch_ids.append(inds)
            
            # Gather features from all processes (distributed training)
            torch.distributed.barrier()
            
            batch_ids = allgather(torch.cat(batch_ids, dim=0), args).squeeze()
            batch_cls = allgather(torch.cat(batch_cls, dim=0), args)
            batch_mask_t = allgather(torch.cat(batch_mask_t, dim=0), args)
            batch_video_feat = allgather(torch.cat(batch_video_feat, dim=0), args)
            batch_mask_v = allgather(torch.cat(batch_mask_v, dim=0), args)
            
            # Reorder based on indices
            batch_cls[batch_ids] = batch_cls.clone()
            batch_mask_t[batch_ids] = batch_mask_t.clone()
            batch_video_feat[batch_ids] = batch_video_feat.clone()
            batch_mask_v[batch_ids] = batch_mask_v.clone()
            
            # Truncate to actual data size
            batch_cls = batch_cls[:batch_ids.max() + 1, ...]
            batch_mask_t = batch_mask_t[:batch_ids.max() + 1, ...]
            batch_video_feat = batch_video_feat[:batch_ids.max() + 1, ...]
            batch_mask_v = batch_mask_v[:batch_ids.max() + 1, ...]
            logger.info('[finish] extract features')
            
            # Stage 2: Calculate similarity matrix
            logger.info('[start] calculate the similarity')
            mini_batch = args.batch_size_val
            sim_matrix = []
            
            batch_cls_split = torch.split(batch_cls, mini_batch)
            batch_mask_t_split = torch.split(batch_mask_t, mini_batch)
            batch_video_feat_split = torch.split(batch_video_feat, mini_batch)
            batch_mask_v_split = torch.split(batch_mask_v, mini_batch)
            
            for cls, text_mask in tqdm(zip(batch_cls_split, batch_mask_t_split), 
                                       total=len(batch_cls_split), desc="Computing similarity"):
                each_row = []
                for video_feat, video_mask in zip(batch_video_feat_split, batch_mask_v_split):
                    logits = model.stage2_eval(cls, text_mask, video_feat, video_mask)
                    logits = logits.cpu().detach().numpy()
                    each_row.append(logits)
                each_row = np.concatenate(tuple(each_row), axis=-1)
                sim_matrix.append(each_row)
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
            logger.info('[finish] calculate the similarity')
        
        # Compute metrics
        logger.info('[start] compute_metrics')
        logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        
        tv_metrics = compute_metrics(sim_matrix)
        vt_metrics = compute_metrics(sim_matrix.T)
        
        logger.info("Text-to-Video (TempMe):")
        logger.info('  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                    format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics.get('R50', 0.0), 
                           tv_metrics['MR'], tv_metrics['MeanR']))
        logger.info("Video-to-Text (TempMe):")
        logger.info('  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$R@50: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                    format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics.get('R50', 0.0),
                           vt_metrics['MR'], vt_metrics['MeanR']))
        
        R1 = tv_metrics['R1']
        return R1
    
    # ========================================================================
    # CLIPKG4Clip Mode: Original evaluation pipeline
    # ========================================================================
    else:
        logger.info("Running CLIPKG4Clip evaluation...")
        
        # #################################################################
        ## below variables are used to multi-sentences retrieval
        # multi_sentence_: important tag for eval
        # cut_off_points: used to tag the label when calculate the metric
        # sentence_num: used to cut the sentence representation
        # video_num: used to cut the video representation
        # #################################################################
        multi_sentence_ = False
        cut_off_points_, sentence_num_, video_num_ = [], -1, -1
        if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
                and test_dataloader.dataset.multi_sentence_per_video:
            multi_sentence_ = True
            cut_off_points_ = test_dataloader.dataset.cut_off_points
            sentence_num_ = test_dataloader.dataset.sentence_num
            video_num_ = test_dataloader.dataset.video_num
            cut_off_points_ = [itm - 1 for itm in cut_off_points_]

        if multi_sentence_:
            logger.warning("Eval under the multi-sentence per video clip setting.")
            logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

        with torch.no_grad():
            batch_list_t = []
            batch_list_v = []
            batch_sequence_output_list, batch_visual_output_list = [], []
            total_video_num = 0

            # ----------------------------
            # 1. cache the features
            # ----------------------------
            for bid, batch in enumerate(test_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, video, video_mask = batch

                if multi_sentence_:
                    # multi-sentences retrieval means: one clip has two or more descriptions.
                    b, *_t = video.shape
                    sequence_output = model.get_sequence_output(input_ids, segment_ids, input_mask)
                    batch_sequence_output_list.append(sequence_output)
                    batch_list_t.append((input_mask, segment_ids,))

                    s_, e_ = total_video_num, total_video_num + b
                    filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                    if len(filter_inds) > 0:
                        video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
                        visual_output = model.get_visual_output(video, video_mask)
                        batch_visual_output_list.append(visual_output)
                        batch_list_v.append((video_mask,))
                    total_video_num += b
                else:
                    sequence_output, visual_output = model.get_sequence_visual_output(input_ids, segment_ids, input_mask, video, video_mask)

                    batch_sequence_output_list.append(sequence_output)
                    batch_list_t.append((input_mask, segment_ids,))

                    batch_visual_output_list.append(visual_output)
                    batch_list_v.append((video_mask,))

                print("{}/{}\r".format(bid, len(test_dataloader)), end="")

            print()  # New line after progress
            logger.info("Finished caching features. Now calculating similarity matrix...")
            
            # ----------------------------------
            # 2. calculate the similarity
            # ----------------------------------
            if n_gpu > 1:
                device_ids = list(range(n_gpu))
                batch_list_t_splits = []
                batch_list_v_splits = []
                batch_t_output_splits = []
                batch_v_output_splits = []
                bacth_len = len(batch_list_t)
                split_len = (bacth_len + n_gpu - 1) // n_gpu
                for dev_id in device_ids:
                    s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
                    if dev_id == 0:
                        batch_list_t_splits.append(batch_list_t[s_:e_])
                        batch_list_v_splits.append(batch_list_v)

                        batch_t_output_splits.append(batch_sequence_output_list[s_:e_])
                        batch_v_output_splits.append(batch_visual_output_list)
                    else:
                        devc = torch.device('cuda:{}'.format(str(dev_id)))
                        devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_t[s_:e_]]
                        batch_list_t_splits.append(devc_batch_list)
                        devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_v]
                        batch_list_v_splits.append(devc_batch_list)

                        devc_batch_list = [b.to(devc) for b in batch_sequence_output_list[s_:e_]]
                        batch_t_output_splits.append(devc_batch_list)
                        devc_batch_list = [b.to(devc) for b in batch_visual_output_list]
                        batch_v_output_splits.append(devc_batch_list)

                parameters_tuple_list = [(batch_list_t_splits[dev_id], batch_list_v_splits[dev_id],
                                          batch_t_output_splits[dev_id], batch_v_output_splits[dev_id]) for dev_id in device_ids]
                parallel_outputs = parallel_apply(_run_on_single_gpu, model, parameters_tuple_list, device_ids)
                sim_matrix = []
                for idx in range(len(parallel_outputs)):
                    sim_matrix += parallel_outputs[idx]
                sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
            else:
                sim_matrix = _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list)
                sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
            
            logger.info("Similarity matrix calculation completed!")

        if multi_sentence_:
            logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
            cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
            max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
            sim_matrix_new = []
            for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
                sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                                      np.full((max_length-e_+s_, sim_matrix.shape[1]), -np.inf)), axis=0))
            sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
            logger.info("after reshape, sim matrix size: {} x {} x {}".
                        format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

            tv_metrics = tensor_text_to_video_metrics(sim_matrix)
            vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
        else:
            logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
            tv_metrics = compute_metrics(sim_matrix)
            vt_metrics = compute_metrics(sim_matrix.T)
            logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

        logger.info("Text-to-Video (CLIPKG4Clip):")
        logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                    format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
        logger.info("Video-to-Text (CLIPKG4Clip):")
        logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                    format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))

        R1 = tv_metrics['R1']
        
        # TempMe mode: Return list of R1 scores for multiple similarity matrices tracking
        if args.use_tempme:
            return [R1]  # List format for compatibility with TempMe's multi-matrix tracking
        else:
            return R1

def _tokenize_text(sentence, tokenizer, max_words, SPECIAL_TOKEN, device):
    """
    Helper function to tokenize text (avoid code duplication).
    Reuses the same logic as dataloader._get_text().
    
    Args:
        sentence (str): Input text
        tokenizer: CLIP tokenizer
        max_words (int): Maximum number of words
        SPECIAL_TOKEN (dict): Special token dictionary
        device: torch device
        
    Returns:
        tuple: (input_ids_tensor, input_mask_tensor, segment_ids_tensor)
    """
    words = tokenizer.tokenize(sentence)
    
    # Add special tokens
    words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
    total_length_with_CLS = max_words - 1
    if len(words) > total_length_with_CLS:
        words = words[:total_length_with_CLS]
    words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]
    
    # Convert to IDs
    input_ids = tokenizer.convert_tokens_to_ids(words)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    
    # Padding
    while len(input_ids) < max_words:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    
    # Convert to tensors
    input_ids_tensor = torch.tensor([input_ids]).to(device)
    input_mask_tensor = torch.tensor([input_mask]).to(device)
    segment_ids_tensor = torch.tensor([segment_ids]).to(device)
    
    return input_ids_tensor, input_mask_tensor, segment_ids_tensor


def eval_epoch_enriched(args, model, test_dataloader, device, n_gpu):
    """
    Enriched evaluation with FQS queries and aggregation strategies.
    Supports both CLIPKG4Clip and TempMe modes.
    
    Pipeline:
    1. Read FQS CSV file (k+1 queries per video)
    2. Extract features for each query INDEPENDENTLY
    3. Compute similarity matrix for each query
    4. Aggregate using Majority Voting or Average Similarity
    5. Compute retrieval metrics
    """
    global logger
    
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)
    
    # Validate CSV file
    if args.val_csv is None or not os.path.exists(args.val_csv):
        logger.error(f"CSV file not found: {args.val_csv}")
        logger.info("Falling back to normal evaluation...")
        return eval_epoch(args, model, test_dataloader, device, n_gpu)
    
    # Load FQS CSV from val_csv
    import pandas as pd
    fqs_df = pd.read_csv(args.val_csv)
    logger.info("="*70)
    logger.info("ENRICHED EVALUATION WITH FQS QUERIES")
    mode_name = "TempMe" if args.use_tempme else "CLIPKG4Clip"
    logger.info(f"Mode: {mode_name}")
    logger.info(f"FQS CSV: {args.val_csv}")
    logger.info(f"Loaded {len(fqs_df)} rows from FQS CSV")
    
    # Determine aggregation strategy - get name from Aggregator class
    aggregator_temp = Aggregator(strategy=args.aggregation_strategy)
    strategy_name = aggregator_temp.strategy_name
    logger.info(f"Aggregation Strategy: {strategy_name}")
    logger.info(f"Expected queries per video: {args.fqs_k + 1}")
    logger.info("="*70)
    
    # Group queries by video_id
    from collections import defaultdict
    video_queries = defaultdict(list)
    
    for _, row in fqs_df.iterrows():
        video_id = row['video_id']
        video_queries[video_id].append({
            'key': row['key'],
            'sentence': row['sentence']
        })
    
    logger.info(f"Grouped into {len(video_queries)} unique videos")
    
    # CRITICAL: Preserve original video order from dataloader (for correct ground truth mapping)
    # DO NOT use sorted() - it will break ground truth alignment!
    all_video_ids_from_loader = test_dataloader.dataset.data['video_id'].values.tolist()
    sorted_video_ids = []  # "sorted" here means order-preserved, not alphabetically sorted!
    seen = set()
    for vid in all_video_ids_from_loader:
        if vid not in seen and vid in video_queries:
            sorted_video_ids.append(vid)
            seen.add(vid)
    
    # Debug: Log first 5 videos to verify order preservation
    logger.info(f"First 5 unique videos (order-preserved): {sorted_video_ids[:5]}")
    
    # Verify each video has expected number of queries
    expected_k = args.fqs_k + 1
    for vid in sorted_video_ids[:3]:
        logger.info(f"  Sample - {vid}: {len(video_queries[vid])} queries")
    
    model.eval()
    
    # ========================================================================
    # TempMe Mode: Use stage1_eval + AllGather + stage2_eval pipeline
    # ========================================================================
    if args.use_tempme:
        logger.info("\nRunning TempMe enriched evaluation...")
        
        # Initialize allgather function
        if AllGather is None:
            raise RuntimeError("AllGather not available. Please check modules.until_module import.")
        allgather = AllGather.apply
        
        tokenizer = ClipTokenizer()
        SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                         "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        
        unique_video_ids = sorted_video_ids
        n_videos = len(unique_video_ids)
        video_id_to_idx = {vid: idx for idx, vid in enumerate(unique_video_ids)}
        
        with torch.no_grad():
            # ========================================================
            # Step 1: Extract video features for UNIQUE videos
            # ========================================================
            logger.info(f"\nStep 1/5: Extracting video features for {n_videos} unique videos (TempMe)...")
            
            batch_video_feat_list = []
            batch_mask_v_list = []
            batch_video_ids = []
            
            processed_videos = 0
            seen_videos = set()
            
            for batch in tqdm(test_dataloader, desc="Extracting video features"):
                batch = tuple(t.to(device) for t in batch)
                text_ids, text_mask, video, video_mask, inds, idx = batch
                
                batch_size = video.shape[0]
                
                # Extract video features using stage1_eval
                _, video_feat = model.stage1_eval(text_ids, text_mask, video, video_mask, idx, shapes=None)
                
                # Process each video in batch
                for i in range(batch_size):
                    # Get video_id from dataloader
                    global_idx = idx[i].item()
                    if global_idx < len(all_video_ids_from_loader):
                        video_id = all_video_ids_from_loader[global_idx]
                        
                        # Only process unique videos
                        if video_id in video_id_to_idx and video_id not in seen_videos:
                            batch_video_feat_list.append(video_feat[i].unsqueeze(0))
                            batch_mask_v_list.append(video_mask[i].unsqueeze(0))
                            batch_video_ids.append(video_id_to_idx[video_id])
                            seen_videos.add(video_id)
                            processed_videos += 1
                
                if processed_videos >= n_videos:
                    break
            
            logger.info(f"Extracted features for {len(batch_video_feat_list)} unique videos")
            
            # ========================================================
            # Step 2: Extract text features for k+1 queries/video
            # ========================================================
            logger.info(f"\nStep 2/5: Extracting text features for {expected_k} queries × {n_videos} videos (TempMe)...")
            
            batch_cls_list = []
            batch_mask_t_list = []
            batch_query_ids = []  # Track which video each query belongs to
            
            for vid_idx, video_id in enumerate(unique_video_ids):
                if video_id not in video_queries:
                    logger.warning(f"Video {video_id} not found in FQS CSV")
                    continue
                
                queries = video_queries[video_id]
                
                # Sort queries by key
                queries = sorted(queries, key=lambda x: (
                    int(x['key'].replace('ret', '').split('_')[0]),
                    int(x['key'].replace('ret', '').split('_')[1]) if '_' in x['key'] else -1
                ))
                
                # Process each of the k+1 queries
                for q_idx, query in enumerate(queries[:expected_k]):
                    sentence = query['sentence']
                    
                    # Tokenize text
                    input_ids, input_mask, segment_ids = _tokenize_text(
                        sentence, tokenizer, args.max_words, SPECIAL_TOKEN, device
                    )
                    
                    # Create dummy video (not used in stage1_eval for text)
                    dummy_video = torch.zeros((1, args.max_frames, 3, 224, 224)).to(device)
                    dummy_video_mask = torch.zeros((1, args.max_frames, 1)).to(device)
                    dummy_idx = torch.tensor([0]).to(device)
                    
                    # Extract text features using stage1_eval
                    cls_feat, _ = model.stage1_eval(input_ids, input_mask, dummy_video, dummy_video_mask, dummy_idx, shapes=None)
                    
                    batch_cls_list.append(cls_feat)
                    batch_mask_t_list.append(input_mask)
                    batch_query_ids.append(vid_idx)
                
                if (vid_idx + 1) % 100 == 0:
                    logger.info(f"Processed {vid_idx + 1}/{n_videos} videos")
            
            logger.info(f"Extracted {len(batch_cls_list)} text features")
            
            # ========================================================
            # Step 3: AllGather (if distributed training)
            # ========================================================
            if args.world_size > 1:
                logger.info("\nStep 3/5: AllGather features from all processes...")
                torch.distributed.barrier()
                
                # Gather video features
                batch_video_ids_tensor = torch.tensor(batch_video_ids, dtype=torch.long, device=device)
                batch_video_feat = torch.cat(batch_video_feat_list, dim=0)
                batch_mask_v = torch.cat(batch_mask_v_list, dim=0)
                
                batch_video_ids_tensor = allgather(batch_video_ids_tensor, args)
                batch_video_feat = allgather(batch_video_feat, args)
                batch_mask_v = allgather(batch_mask_v, args)
                
                # Reorder video features
                batch_video_feat[batch_video_ids_tensor] = batch_video_feat.clone()
                batch_mask_v[batch_video_ids_tensor] = batch_mask_v.clone()
                
                # Truncate to actual size
                batch_video_feat = batch_video_feat[:n_videos, ...]
                batch_mask_v = batch_mask_v[:n_videos, ...]
                
                # Gather text features
                batch_query_ids_tensor = torch.tensor(batch_query_ids, dtype=torch.long, device=device)
                batch_cls = torch.cat(batch_cls_list, dim=0)
                batch_mask_t = torch.cat(batch_mask_t_list, dim=0)
                
                batch_query_ids_tensor = allgather(batch_query_ids_tensor, args)
                batch_cls = allgather(batch_cls, args)
                batch_mask_t = allgather(batch_mask_t, args)
                
                # Reorder text features (not needed since we use sequential order)
                # Truncate to actual size
                batch_cls = batch_cls[:len(batch_cls_list), ...]
                batch_mask_t = batch_mask_t[:len(batch_mask_t_list), ...]
                
                logger.info(f"AllGather complete: video {batch_video_feat.shape}, text {batch_cls.shape}")
            else:
                # Single GPU: just concatenate
                batch_video_feat = torch.cat(batch_video_feat_list, dim=0)
                batch_mask_v = torch.cat(batch_mask_v_list, dim=0)
                batch_cls = torch.cat(batch_cls_list, dim=0)
                batch_mask_t = torch.cat(batch_mask_t_list, dim=0)
            
            # ========================================================
            # Step 4: Compute k+1 similarity matrices using stage2_eval
            # ========================================================
            logger.info(f"\nStep 4/5: Computing {expected_k} similarity matrices (TempMe)...")
            
            sim_matrices_list = []
            mini_batch = args.batch_size_val
            
            for k_idx in range(expected_k):
                # Extract text features for this query type (every k+1-th element)
                k_batch_cls = batch_cls[k_idx::expected_k]
                k_batch_mask_t = batch_mask_t[k_idx::expected_k]
                
                # Compute similarity matrix
                sim_matrix = []
                
                k_batch_cls_split = torch.split(k_batch_cls, mini_batch)
                k_batch_mask_t_split = torch.split(k_batch_mask_t, mini_batch)
                batch_video_feat_split = torch.split(batch_video_feat, mini_batch)
                batch_mask_v_split = torch.split(batch_mask_v, mini_batch)
                
                for cls, text_mask in zip(k_batch_cls_split, k_batch_mask_t_split):
                    each_row = []
                    for video_feat, video_mask in zip(batch_video_feat_split, batch_mask_v_split):
                        # Use stage2_eval to compute similarity
                        b1b2_logits = model.stage2_eval(cls, video_feat, text_mask, video_mask, shapes=None)
                        b1b2_logits = b1b2_logits.cpu().detach().numpy()
                        each_row.append(b1b2_logits)
                    each_row = np.concatenate(tuple(each_row), axis=-1)
                    sim_matrix.append(each_row)
                
                k_sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
                sim_matrices_list.append(k_sim_matrix)
                logger.info(f"  Query {k_idx+1}/{expected_k}: shape {k_sim_matrix.shape}")
            
            # ========================================================
            # Step 5: Aggregate using selected strategy
            # ========================================================
            logger.info(f"\nStep 5/5: Aggregating using {strategy_name}...")
            
            aggregator = Aggregator(strategy=args.aggregation_strategy)
            sim_matrices_array = np.stack(sim_matrices_list, axis=0)
            logger.info(f"Stacked similarity matrices shape: {sim_matrices_array.shape}")
            
            final_sim_matrix = aggregator.aggregate(sim_matrices_array)
            logger.info(f"Final aggregated similarity matrix shape: {final_sim_matrix.shape}")
    
    # ========================================================================
    # CLIPKG4Clip Mode: Original implementation
    # ========================================================================
    else:
        logger.info("\nRunning CLIPKG4Clip enriched evaluation...")
        
        tokenizer = ClipTokenizer()
        
        # Define CLIP special tokens (MUST use exact same as dataloader)
        SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                         "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        
        with torch.no_grad():
            # ========================================================
            # Step 1: Extract video features for UNIQUE videos only
            # ========================================================
            logger.info("\nStep 1/4: Extracting video features for unique videos (CLIPKG4Clip)...")
            
            # Use sorted_video_ids (already computed earlier from all_video_ids_from_loader)
            unique_video_ids = sorted_video_ids
            n_videos = len(unique_video_ids)
            logger.info(f"Total rows in CSV: {len(all_video_ids_from_loader)}")
            logger.info(f"Unique videos to process: {n_videos}")
            logger.info(f"First 3 unique video IDs: {unique_video_ids[:3]}")
            
            # Create video_id -> feature index mapping
            video_id_to_idx = {vid: idx for idx, vid in enumerate(unique_video_ids)}
            
            # Extract features for unique videos only
            batch_visual_output_list = []
            batch_list_v = []
            
            processed_videos = 0
            row_idx = 0  # Track current row index in CSV
            seen = set()  # Reset seen set for duplicate detection during extraction
            
            for bid, batch in enumerate(test_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, video, video_mask = batch
                
                # batch contains multiple videos (batch_size_val)
                batch_size = video.shape[0]
                
                # Extract visual features for entire batch at once (same as eval_epoch)
                visual_output = model.get_visual_output(video, video_mask)
                
                # Process each video in the batch
                for i in range(batch_size):
                    current_video_id = all_video_ids_from_loader[row_idx]
                    row_idx += 1
                    
                    # Only keep if this is the FIRST occurrence of this video
                    if current_video_id in seen:
                        continue  # Skip duplicate
                    
                    # Mark as seen
                    seen.add(current_video_id)
                    
                    # Extract single video features from batch
                    single_visual_output = visual_output[i:i+1]  # Keep batch dimension
                    single_video_mask = video_mask[i:i+1]
                    
                    batch_visual_output_list.append(single_visual_output)
                    batch_list_v.append((single_video_mask,))
                    processed_videos += 1
                    
                    if processed_videos % 100 == 0 or processed_videos == n_videos:
                        print(f"Processed {processed_videos}/{n_videos} unique videos\r", end="")
                    
                    # Early stop if we've processed all unique videos
                    if processed_videos >= n_videos:
                        break
                
                if processed_videos >= n_videos:
                    break
            
            print()  # New line
            logger.info(f"Extracted features for {len(batch_visual_output_list)} unique videos")
            
            # ========================================================
            # Step 2: Extract text features for k+1 queries/video
            # Process queries in order of UNIQUE videos
            # ========================================================
            logger.info(f"\nStep 2/4: Extracting text features for {expected_k} queries/video (CLIPKG4Clip)...")
            batch_sequence_output_list = []  # Will store text features for ALL queries
            batch_list_t = []
            
            for vid_idx, video_id in enumerate(unique_video_ids):
                if video_id not in video_queries:
                    logger.error(f"Video {video_id} not found in FQS CSV!")
                    continue
                    
                queries = video_queries[video_id]
                
                # Sort queries by key: ret0, ret0_1, ret0_2, ...
                queries = sorted(queries, key=lambda x: (
                    int(x['key'].replace('ret', '').split('_')[0]),
                    int(x['key'].replace('ret', '').split('_')[1]) if '_' in x['key'] else -1
                ))
                
                # Process each of the k+1 queries
                for q_idx, query in enumerate(queries[:expected_k]):
                    sentence = query['sentence']
                    words = tokenizer.tokenize(sentence)
                    
                    # Use CLIP special tokens (SAME as dataloader)
                    words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
                    total_length_with_CLS = args.max_words - 1
                    if len(words) > total_length_with_CLS:
                        words = words[:total_length_with_CLS]
                    words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]
                    
                    input_ids = tokenizer.convert_tokens_to_ids(words)
                    input_mask = [1] * len(input_ids)
                    segment_ids = [0] * len(input_ids)
                    
                    # Padding
                    while len(input_ids) < args.max_words:
                        input_ids.append(0)
                        input_mask.append(0)
                        segment_ids.append(0)
                    
                    input_ids_tensor = torch.tensor([input_ids]).to(device)
                    input_mask_tensor = torch.tensor([input_mask]).to(device)
                    segment_ids_tensor = torch.tensor([segment_ids]).to(device)
                    
                    # Extract text features
                    sequence_output = model.get_sequence_output(input_ids_tensor, segment_ids_tensor, input_mask_tensor)
                    batch_sequence_output_list.append(sequence_output)
                    batch_list_t.append((input_mask_tensor, segment_ids_tensor,))
                
                if (vid_idx + 1) % 100 == 0:
                    logger.info(f"  Processed {vid_idx + 1}/{n_videos} videos")
            
            logger.info(f"Extracted {len(batch_sequence_output_list)} text features ({expected_k} queries × {n_videos} videos)")
            
            # ========================================================
            # Step 3: Compute k+1 INDEPENDENT similarity matrices
            # Using SAME approach as eval_epoch (reuse _run_on_single_gpu)
            # ========================================================
            logger.info(f"\nStep 3/4: Computing {expected_k} INDEPENDENT similarity matrices (CLIPKG4Clip)...")
            
            sim_matrices_list = []  # Will contain k+1 matrices, each (n_videos, n_videos)
            
            for k_idx in range(expected_k):
                # Extract batch lists for this query type (every k+1-th element starting from k_idx)
                k_batch_list_t = []
                k_batch_sequence_output_list = []
                
                for vid_idx in range(n_videos):
                    query_idx = vid_idx * expected_k + k_idx
                    k_batch_sequence_output_list.append(batch_sequence_output_list[query_idx])
                    k_batch_list_t.append(batch_list_t[query_idx])
                
                # Compute similarity matrix using SAME method as eval_epoch
                if n_gpu > 1:
                    device_ids = list(range(n_gpu))
                    k_batch_list_t_splits = []
                    k_batch_sequence_output_splits = []
                    batch_list_v_splits = []
                    batch_visual_output_splits = []
                    batch_len = len(k_batch_list_t)
                    split_len = (batch_len + n_gpu - 1) // n_gpu
                    for dev_id in device_ids:
                        s_ = dev_id * split_len
                        e_ = min(s_ + split_len, batch_len)
                        if s_ < batch_len:
                            k_batch_list_t_splits.append(k_batch_list_t[s_:e_])
                            k_batch_sequence_output_splits.append(k_batch_sequence_output_list[s_:e_])
                            
                            batch_list_v_splits.append(batch_list_v)
                            batch_visual_output_splits.append(batch_visual_output_list)
                    
                    parameters_tuple_list = [(model, k_batch_list_t_splits[dev_id], batch_list_v_splits[dev_id],
                                              k_batch_sequence_output_splits[dev_id], batch_visual_output_splits[dev_id]) for dev_id in device_ids]
                    parallel_outputs = parallel_apply(_run_on_single_gpu, model, parameters_tuple_list, device_ids)
                    k_sim_matrix = []
                    for idx in range(len(parallel_outputs)):
                        k_sim_matrix.extend(parallel_outputs[idx])
                    k_sim_matrix = np.concatenate(tuple(k_sim_matrix), axis=0)
                else:
                    k_sim_matrix = _run_on_single_gpu(model, k_batch_list_t, batch_list_v, 
                                                      k_batch_sequence_output_list, batch_visual_output_list)
                    k_sim_matrix = np.concatenate(tuple(k_sim_matrix), axis=0)
                
                sim_matrices_list.append(k_sim_matrix)
                logger.info(f"  Query {k_idx+1}/{expected_k}: Computed similarity matrix shape {k_sim_matrix.shape}")
            
            # Step 4: Aggregate using selected strategy
            logger.info(f"\nStep 4/4: Aggregating using {strategy_name}...")
            
            # Create Aggregator
            aggregator = Aggregator(strategy=args.aggregation_strategy)
            
            # Stack into (k+1, n_videos, n_videos)
            sim_matrices_array = np.stack(sim_matrices_list, axis=0)
            logger.info(f"Stacked similarity matrices shape: {sim_matrices_array.shape}")
            
            # Aggregate
            final_sim_matrix = aggregator.aggregate(sim_matrices_array)
            
            logger.info(f"Final aggregated similarity matrix shape: {final_sim_matrix.shape}")
    
    # Compute metrics
    logger.info("\n" + "="*70)
    logger.info("COMPUTING RETRIEVAL METRICS")
    logger.info("="*70)
    
    tv_metrics = compute_metrics(final_sim_matrix)
    vt_metrics = compute_metrics(final_sim_matrix.T)
    
    logger.info(f'Length-T: {len(final_sim_matrix)}, Length-V: {len(final_sim_matrix[0])}')
    
    logger.info(f"\nText-to-Video ({strategy_name}):")
    logger.info('  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info(f"\nVideo-to-Text ({strategy_name}):")
    logger.info('  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))
    
    logger.info("="*70)
    
    R1 = tv_metrics['R1']
    return R1

def main():
    global logger
    
    # TempMe: Initialize global variables for tracking best scores across multiple similarity matrices
    global best_score, best_score_list, meters, sim_matrix_num, sim_name_list
    
    args = get_args()
    args = set_seed_logger(args)
    
    # ========================================================================
    # TempMe Mode: Initialize global variables and args mapping
    # ========================================================================
    # These are used in train_epoch() for periodic evaluation
    sim_name_list = ['base']
    sim_matrix_num = len(sim_name_list)
    best_score = 0.00001
    best_score_list = [0.00001 for _ in range(sim_matrix_num)]
    meters = None
    
    # ========================================================================
    # TempMe Mode: Additional initialization and args mapping
    # ========================================================================
    if args.use_tempme:
        meters = MetricLogger(delimiter="  ") if METRICLOGGER_AVAILABLE else None
        logger.info("[TempMe] Initialized global tracking variables")
        
        # Log TempMe defaults that were set in get_args()
        logger.info(f"[TempMe] tome_r = {args.tome_r}")
        logger.info(f"[TempMe] lora_dim = {args.lora_dim}")
        logger.info(f"[TempMe] frame_pos = {args.frame_pos}")
        
        # Args mapping:
        # Critical: VTRModel uses args.base_encoder instead of args.pretrained_clip_name
        if args.base_encoder is None:
            args.base_encoder = args.pretrained_clip_name
            logger.info(f"[TempMe] Mapped args.base_encoder = '{args.base_encoder}' (from pretrained_clip_name)")
        else:
            logger.info(f"[TempMe] Using user-provided args.base_encoder = '{args.base_encoder}'")
        
        # Critical: TempMe dataloaders use args.video_framerate instead of args.feature_framerate
        if args.video_framerate is None:
            args.video_framerate = args.feature_framerate if hasattr(args, 'feature_framerate') else 1
            logger.info(f"[TempMe] Mapped args.video_framerate = {args.video_framerate} (from feature_framerate)")
        else:
            logger.info(f"[TempMe] Using user-provided args.video_framerate = {args.video_framerate}")
        
        # Ensure device is set correctly (TempMe expects string 'cuda' or torch.device)
        if args.device is None or args.device == '':
            args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", args.local_rank)
            logger.info(f"[TempMe] Set args.device = {args.device}")
        elif isinstance(args.device, str):
            # Convert string to torch.device
            if args.device == 'cuda' and torch.cuda.is_available():
                args.device = torch.device("cuda", args.local_rank)
            else:
                args.device = torch.device(args.device)
            logger.info(f"[TempMe] Converted args.device to {args.device}")
        
        # Map distributed flag (TempMe uses different convention)
        if args.distributed == 0 and args.world_size > 1:
            args.distributed = 1
            logger.info(f"[TempMe] Set args.distributed = 1 (world_size > 1)")
    else:
        # CLIPKG4Clip mode: Ensure backward compatibility
        if args.base_encoder is None:
            args.base_encoder = args.pretrained_clip_name  # Set for consistency
        if args.video_framerate is None:
            args.video_framerate = args.feature_framerate if hasattr(args, 'feature_framerate') else 1
        if args.device is None:
            args.device = "cuda"  # CLIPKG4Clip doesn't use this, but set for safety
    
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = ClipTokenizer()

    assert  args.task_type == "retrieval"
    model = init_model(args, device, n_gpu, args.local_rank)

    ## ####################################
    # freeze testing
    ## ####################################
    assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
    if hasattr(model, "clip") and args.freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():
            # top layers always need to train
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                    or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                continue    # need to train
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num:
                    continue    # need to train

            if args.linear_patch == "3d" and name.find("conv2."):
                continue
            else:
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False

    ## ####################################
    # dataloader loading
    ## ####################################
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None \
           or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    # For enriched evaluation: only need 1 dataloader (test or val) to load videos
    # Queries will be read from CSV file (--val_csv)
    if args.do_eval and args.eval_enriched == 1:
        # If test_dataloader is None, load val_dataloader and use it
        if test_dataloader is None:
            if DATALOADER_DICT[args.datatype]["val"] is not None:
                test_dataloader, test_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
            else:
                raise ValueError("No dataloader available for enriched evaluation")
        # Don't need val_dataloader for enriched eval
        val_dataloader, val_length = None, 0
    else:
        # Normal evaluation: need both test and val dataloaders
        if DATALOADER_DICT[args.datatype]["val"] is not None:
            val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
        else:
            val_dataloader, val_length = test_dataloader, test_length

        ## report validation results if the ["test"] is None
        if test_dataloader is None:
            test_dataloader, test_length = val_dataloader, val_length

    if args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        if test_dataloader is not None:
            logger.info("  Num steps = %d", len(test_dataloader))
        if not (args.do_eval and args.eval_enriched == 1) and val_dataloader is not None:
            logger.info("***** Running val *****")
            logger.info("  Num examples = %d", val_length)

    ## ####################################
    # train and eval
    ## ####################################
    if args.do_train:
        # Check if we need to do enriched data training first
        enriched_checkpoint = None
        use_enriched_training = False
        
        # For MSRVTT: use enriched_data_path
        if args.datatype == "msrvtt" and args.enriched_data_path is not None:
            if os.path.exists(args.enriched_data_path):
                use_enriched_training = True
                if args.local_rank == 0:
                    logger.info(f"[ENRICHED] Found enriched data at: {args.enriched_data_path}")
            else:
                if args.local_rank == 0:
                    logger.warning(f"[ENRICHED] Enriched data path specified but NOT FOUND: {args.enriched_data_path}")
                    logger.warning(f"[ENRICHED] Skipping enriched training and proceeding with original data only.")
        # For MSVD: use enriched flag
        elif args.datatype == "msvd" and args.enriched == "yes":
            use_enriched_training = True
            if args.local_rank == 0:
                logger.info(f"[ENRICHED] Using enriched data for MSVD (enriched={args.enriched})")
        
        if use_enriched_training:
            if args.local_rank == 0:
                logger.info("="*50)
                logger.info("STAGE 1: Training on ENRICHED DATA")
                logger.info("="*50)
                if args.use_tempme:
                    logger.info("Mode: TempMe (VTRModel with LoRA + ToMe)")
                else:
                    logger.info("Mode: CLIPKG4Clip (Original)")
                if args.datatype == "msrvtt":
                    logger.info("Enriched data path: %s", args.enriched_data_path)
                else:
                    logger.info("Using enriched captions for MSVD")
                logger.info("Enriched epochs: %d", args.enriched_epochs)
                logger.info("Enriched max steps: %d", args.enriched_max_steps)
            
            # Temporarily swap parameters
            # Save original values for restoration after enriched training
            original_data_path = args.data_path
            original_anno_path = args.anno_path if hasattr(args, 'anno_path') else None
            original_anno_json_name = getattr(args, 'anno_json_name', 'MSRVTT_data.json')
            original_epochs = args.epochs
            original_max_steps = args.max_steps
            original_enriched = None
            
            if args.datatype == "msrvtt":
                # TempMe vs CLIPKG4Clip use different data loading mechanisms
                if args.use_tempme:
                    # TempMe mode: Uses anno_json_name parameter to specify JSON filename
                    # Keep anno_path as folder, change only JSON filename
                    # Always extract just the filename to avoid path duplication
                    args.anno_json_name = os.path.basename(args.enriched_data_path)
                    if args.local_rank == 0:
                        logger.info("[TempMe] Switched anno_json_name to: %s", args.anno_json_name)
                else:
                    # CLIPKG4Clip mode: Uses data_path (direct path to pickle file)
                    # enriched_data_path should point to enriched pickle file
                    args.data_path = args.enriched_data_path
                    if args.local_rank == 0:
                        logger.info("[CLIPKG4Clip] Switched data_path to: %s", args.data_path)
            elif args.datatype == "msvd":
                # For MSVD, keep data_path but mark as using enriched
                original_enriched = args.enriched
                args.enriched = "yes"
                if args.local_rank == 0:
                    logger.info("[MSVD] Switched enriched flag to: yes")
            
            args.epochs = args.enriched_epochs
            args.max_steps = args.enriched_max_steps
            
            # Load enriched data
            train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
            num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                            / args.gradient_accumulation_steps) * args.epochs

            coef_lr = args.coef_lr
            optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

            if args.local_rank == 0:
                logger.info("***** Running ENRICHED data training *****")
                logger.info("  Num examples = %d", train_length)
                logger.info("  Batch size = %d", args.batch_size)
                logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

            last_enriched_model_file = "None"
            
            global_step = 0
            for epoch in range(0, args.enriched_epochs):
                train_sampler.set_epoch(epoch)
                tr_loss, global_step, early_stop = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                                   scheduler, global_step, local_rank=args.local_rank, val_dataloader=val_dataloader)
                if args.local_rank == 0:
                    logger.info("[ENRICHED] Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.enriched_epochs, tr_loss)

                    # Save checkpoint for this epoch
                    output_model_file = save_model(epoch, args, model, optimizer, tr_loss, type_name="enriched")
                    last_enriched_model_file = output_model_file
                
                # Stop if reached max_steps
                if early_stop:
                    break
            
            # Use the last enriched checkpoint for next stage
            enriched_checkpoint = last_enriched_model_file
            
            # Restore original parameters for STAGE 2 training
            args.data_path = original_data_path
            args.epochs = original_epochs
            args.max_steps = original_max_steps
            
            # Restore mode-specific parameters
            if args.use_tempme:
                # TempMe mode: Restore original anno_json_name
                args.anno_json_name = original_anno_json_name
                if args.local_rank == 0:
                    logger.info("[TempMe] Restored anno_json_name to: %s", args.anno_json_name)
            
            if args.datatype == "msvd" and original_enriched is not None:
                args.enriched = "no"  # Switch to raw captions for MSVD
                if args.local_rank == 0:
                    logger.info("[MSVD] Restored enriched flag to: no")
            
            if args.local_rank == 0:
                logger.info("="*50)
                logger.info("STAGE 1 COMPLETED: Using last checkpoint: %s", enriched_checkpoint)
                logger.info("="*50)
        
        # STAGE 2: Train on original data
        if args.local_rank == 0:
            logger.info("="*50)
            logger.info("STAGE 2: Training on ORIGINAL DATA")
            logger.info("="*50)
        
        # Load original data
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        # If we have enriched checkpoint, load it as init model
        if enriched_checkpoint is not None:
            if args.local_rank == 0:
                logger.info("Loading enriched checkpoint: %s", enriched_checkpoint)
            # Load the enriched model
            model = load_model(-1, args, n_gpu, device, model_file=enriched_checkpoint)
        
        coef_lr = args.coef_lr
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

        if args.local_rank == 0:
            logger.info("***** Running ORIGINAL data training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = 0.00001
        best_output_model_file = "None"
        ## ##############################################################
        # resume optimizer state besides loss to continue train
        ## ##############################################################
        resumed_epoch = 0
        if args.resume_model:
            checkpoint = torch.load(args.resume_model, map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            resumed_epoch = checkpoint['epoch']+1
            resumed_loss = checkpoint['loss']
        
        global_step = 0
        for epoch in range(resumed_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            tr_loss, global_step, early_stop = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                               scheduler, global_step, local_rank=args.local_rank, val_dataloader=val_dataloader)
            if args.local_rank == 0:
                logger.info("[ORIGINAL] Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)

                output_model_file = save_model(epoch, args, model, optimizer, tr_loss, type_name="")

                ## Run on val dataset, this process is *TIME-consuming*.
                # logger.info("Eval on val dataset")
                # R1 = eval_epoch(args, model, val_dataloader, device, n_gpu)

                R1 = eval_epoch(args, model, test_dataloader, device, n_gpu)
                if best_score <= R1:
                    best_score = R1
                    best_output_model_file = output_model_file
                logger.info("[ORIGINAL] The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))
            
            # Dừng training nếu đã đạt max_steps
            if early_stop:
                break

        ## Uncomment if want to test on the best checkpoint
        # if args.local_rank == 0:
        #     model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file)
        #     eval_epoch(args, model, test_dataloader, device, n_gpu)

    elif args.do_eval:
        if args.local_rank == 0:
            # Check evaluation mode
            if args.eval_enriched == 1:
                logger.info("Using ENRICHED evaluation with FQS queries")
                eval_epoch_enriched(args, model, test_dataloader, device, n_gpu)
            else:
                logger.info("Using NORMAL evaluation (original queries only)")
                eval_epoch(args, model, test_dataloader, device, n_gpu)

if __name__ == "__main__":
    main()
