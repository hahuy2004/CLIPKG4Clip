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
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4Clip
from modules.optimization import BertAdam

from util import parallel_apply, get_logger
from dataloaders.data_dataloaders import DATALOADER_DICT

# Import enriched evaluation modules
from enriched_eval.fqs_selector import farthest_query_selection
from enriched_eval.aggregator import Aggregator

torch.distributed.init_process_group(backend="nccl")

global logger

def get_args(description='CLIP4Clip on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument('--train_csv', type=str, default='data/.train.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')
    
    # Enriched data training parameters
    parser.add_argument('--enriched_data_path', type=str, default=None, help='Path to enriched captions JSON file for pre-training (MSRVTT)')
    parser.add_argument('--enriched', type=str, default='no', choices=['yes', 'no'], help='Use enriched captions for MSVD dataset')
    parser.add_argument('--enriched_epochs', type=int, default=3, help='Number of epochs to train on enriched data')
    parser.add_argument('--enriched_max_steps', type=int, default=-1, help='Max training steps for enriched data, -1 means no limit')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
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

    # FQS evaluation parameters
    parser.add_argument('--fqs_csv_path', type=str, default=None,
                        help='Path to FQS CSV file (e.g., MSRVTT_JSFUSION_test_fqs.csv)')
    parser.add_argument('--eval_enriched', type=int, default=0, choices=[0, 1],
                        help='Use enriched queries: 0=no (only original), 1=yes (use FQS queries)')
    parser.add_argument('--aggregation_strategy', type=int, default=1, choices=[1, 2],
                        help='Aggregation strategy: 1=Majority Voting, 2=Average Similarity (only when eval_enriched=1)')
    parser.add_argument('--fqs_k', type=int, default=2,
                        help='Number of enriched queries per video (default: 2, total k+1=3 queries)')

    args = parser.parse_args()

    if args.sim_header == "tightTransf":
        args.loose_type = False

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

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
    
    if args.eval_enriched == 1 and args.fqs_csv_path is None:
        raise ValueError(
            "ERROR: --fqs_csv_path is required when --eval_enriched=1. "
            "Please provide the path to FQS CSV file."
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

    if hasattr(model, 'module'):
        model = model.module

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

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=False)

    return optimizer, scheduler, model

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

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    early_stop = False

    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

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

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

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

    model.eval()
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

    logger.info("Text-to-Video:")
    logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info("Video-to-Text:")
    logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))

    R1 = tv_metrics['R1']
    return R1

def eval_epoch_enriched(args, model, test_dataloader, device, n_gpu):
    """
    Enriched evaluation with FQS queries and aggregation strategies.
    
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
    
    # Validate FQS CSV file
    if args.fqs_csv_path is None or not os.path.exists(args.fqs_csv_path):
        logger.error(f"FQS CSV file not found: {args.fqs_csv_path}")
        logger.info("Falling back to normal evaluation...")
        return eval_epoch(args, model, test_dataloader, device, n_gpu)
    
    # Load FQS CSV
    import pandas as pd
    fqs_df = pd.read_csv(args.fqs_csv_path)
    logger.info("="*70)
    logger.info("ENRICHED EVALUATION WITH FQS QUERIES")
    logger.info(f"FQS CSV: {args.fqs_csv_path}")
    logger.info(f"Loaded {len(fqs_df)} rows from FQS CSV")
    
    # Determine aggregation strategy
    strategy_name = "Majority Voting" if args.aggregation_strategy == 1 else "Average Similarity"
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
    
    # Sort video IDs for consistent ordering
    sorted_video_ids = sorted(video_queries.keys())
    
    # Verify each video has expected number of queries
    expected_k = args.fqs_k + 1
    for vid in sorted_video_ids[:3]:
        logger.info(f"  Sample - {vid}: {len(video_queries[vid])} queries")
    
    model.eval()
    tokenizer = ClipTokenizer()
    
    with torch.no_grad():
        # Step 1: Extract video features (once)
        logger.info("\nStep 1/4: Extracting video features...")
        video_features = []
        
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, video, video_mask = batch
            
            # Get video features
            visual_output = model.get_visual_output(video, video_mask)
            video_features.append(visual_output.cpu())
            
            if (bid + 1) % 50 == 0:
                logger.info(f"  Processed {bid + 1}/{len(test_dataloader)} video batches")
        
        video_features = torch.cat(video_features, dim=0)  # (n_videos, dim)
        n_videos = video_features.shape[0]
        logger.info(f"Extracted features for {n_videos} videos, shape: {video_features.shape}")
        
        # Step 2: Extract text features for k+1 queries per video
        logger.info(f"\nStep 2/4: Extracting text features for {expected_k} queries/video...")
        
        all_text_features = []  # Will be list of (k+1, dim) for each video
        
        for vid_idx, video_id in enumerate(sorted_video_ids):
            queries = video_queries[video_id]
            
            # Sort queries by key: ret0, ret0_1, ret0_2, ...
            queries = sorted(queries, key=lambda x: (
                int(x['key'].replace('ret', '').split('_')[0]),
                int(x['key'].replace('ret', '').split('_')[1]) if '_' in x['key'] else -1
            ))
            
            # Tokenize and extract features for each query
            video_text_features = []
            for query in queries[:expected_k]:
                sentence = query['sentence']
                words = tokenizer.tokenize(sentence)
                words = ["[CLS]"] + words[:args.max_words - 2] + ["[SEP]"]
                
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
                video_text_features.append(sequence_output.cpu())
            
            # Stack k+1 queries: (k+1, dim)
            video_text_features = torch.cat(video_text_features, dim=0)
            all_text_features.append(video_text_features)
            
            if (vid_idx + 1) % 100 == 0:
                logger.info(f"  Processed {vid_idx + 1}/{len(sorted_video_ids)} videos")
        
        # Stack all: (n_videos, k+1, dim)
        all_text_features = torch.stack(all_text_features, dim=0)
        logger.info(f"Text features shape: {all_text_features.shape}")
        
        # Step 3: Compute INDEPENDENT similarity matrices for each query type
        logger.info(f"\nStep 3/4: Computing {expected_k} INDEPENDENT similarity matrices...")
        
        sim_matrices_list = []  # Will contain k+1 matrices, each (n_videos, n_videos)
        
        for k_idx in range(expected_k):
            # Get text features for k-th query across all videos: (n_videos, dim)
            text_k = all_text_features[:, k_idx, :]
            
            # Move to device
            text_k = text_k.to(device)
            video_feat = video_features.to(device)
            
            # Normalize for cosine similarity
            text_k_norm = text_k / text_k.norm(dim=-1, keepdim=True)
            video_feat_norm = video_feat / video_feat.norm(dim=-1, keepdim=True)
            
            # Compute similarity matrix: (n_videos, n_videos)
            sim_matrix = torch.matmul(text_k_norm, video_feat_norm.T)
            sim_matrices_list.append(sim_matrix.cpu().numpy())
            
            logger.info(f"  Similarity matrix {k_idx + 1}/{expected_k}: {sim_matrix.shape}")
        
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
    args = get_args()
    args = set_seed_logger(args)
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
        logger.info("  Num steps = %d", len(test_dataloader))
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
        if args.datatype == "msrvtt" and args.enriched_data_path is not None and os.path.exists(args.enriched_data_path):
            use_enriched_training = True
        # For MSVD: use enriched flag
        elif args.datatype == "msvd" and args.enriched == "yes":
            use_enriched_training = True
        
        if use_enriched_training:
            if args.local_rank == 0:
                logger.info("="*50)
                logger.info("STAGE 1: Training on ENRICHED DATA")
                logger.info("="*50)
                if args.datatype == "msrvtt":
                    logger.info("Enriched data path: %s", args.enriched_data_path)
                else:
                    logger.info("Using enriched captions for MSVD")
                logger.info("Enriched epochs: %d", args.enriched_epochs)
                logger.info("Enriched max steps: %d", args.enriched_max_steps)
            
            # Temporarily swap parameters
            original_data_path = args.data_path
            original_epochs = args.epochs
            original_max_steps = args.max_steps
            original_enriched = None
            
            if args.datatype == "msrvtt":
                args.data_path = args.enriched_data_path
            elif args.datatype == "msvd":
                # For MSVD, keep data_path but mark as using enriched
                original_enriched = args.enriched
                args.enriched = "yes"
            
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
                                                   scheduler, global_step, local_rank=args.local_rank)
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
            
            # Restore original parameters
            args.data_path = original_data_path
            args.epochs = original_epochs
            args.max_steps = original_max_steps
            if args.datatype == "msvd" and original_enriched is not None:
                args.enriched = "no"  # Switch to raw captions for MSVD
            
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
                                               scheduler, global_step, local_rank=args.local_rank)
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
