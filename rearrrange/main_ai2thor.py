import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from rearrange_on_proc.rearrange_challenge.constants import OTHER_INFO_DIR
from rearrange_on_proc.arguments import args
from rearrange_on_proc.tasks_ai2thor import RearrangementTask_AI2THOR
from rearrange_on_proc.offline_ai2thor.environment import Environment
import prior
import os
import sys
import json
import torch.multiprocessing as mp
import platform
import torch
import time
import compress_pickle
import itertools
import logging
import queue
from setproctitle import setproctitle as ptitle
from constants import DATASET_DIR
from operator import itemgetter
from ai2thor.controller import Controller
from allenact.utils.misc_utils import NumpyJSONEncoder
import warnings
warnings.filterwarnings('ignore')
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
# available_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
import ssl
 



def load_dataset_from_path(split):
    # data_path = os.path.abspath(os.path.join(DATASET_DIR, f"{split}_0424.pkl.gz"))
    data_path = "/home/lyy/rearrange_on_ai2thor/rearrange_on_proc/data/2022/train.pkl.gz"
    # data_path = os.path.abspath(os.path.join(
    #     DATASET_DIR, f"debug_add33.pkl.gz"))
    if not os.path.exists(data_path):
        raise RuntimeError(f"No data at path {data_path}")

    print(f'Loading {split} dataset...')
    data = compress_pickle.load(path=data_path)
    task_id_to_task = {}
    for scene in data.keys():
        scene_tasks = data[scene]
        for i in range(len(scene_tasks)):
            task_id = f"{scene}_{i}"
            task_id_to_task[task_id] = scene_tasks[i]

        
    return task_id_to_task


def run_rearrange_task(task_id,controller, task, process_id, task_index, solution_config, online_controller,gpu_id):
    aithor_rearrangement = RearrangementTask_AI2THOR(task_id,controller, task, process_id, task_index, solution_config,gpu_id, online_controller)
    task_id, metrics, explore_time, all_time  = aithor_rearrangement.main()
    return task_id, metrics, explore_time, all_time

def rearrange_worker(
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    process_id,
    solution_config,
    x_display,
    gpu_id
):
    task_index = 0
    # device = torch.device(f"cuda:{process_id % available_devices_num}") if available_devices_num > 0 and torch.cuda.is_available(
    # ) else torch.device('cpu')
    # x_display_id = process_id % available_devices_num if available_devices_num > 0  else 0

    online_controller = Controller(
                            x_display=x_display,
                            width=args.W,
                            height=args.H,
                            fieldOfView=args.fov,
                            gridSize=args.STEP_SIZE,
                            rotateStepDegrees=args.DT,
                            renderDepthImage=True,
                            renderInstanceSegmentation=True,
                            commit_id='eb93d0b6520e567bac8ad630462b5c0c4cea1f5f',
                            # snapToGrid = False,
                            )
    if args.use_offline_ai2thor:
        controller = Environment(
            use_offline_controller=True,
            fov = args.fov,
            offline_data_dir = '/home/yyl/lwj/rearrange_on_ProcTHOR/rearrange_on_proc/data/offline',
        )
    else:
        controller = online_controller

    while True:
        try:
            task_id,task= input_queue.get(timeout=2)
            task_index += 1
        except queue.Empty:
            break

        task_id, metrics, explore_time, all_time = run_rearrange_task(
            task_id = task_id,
            controller=controller,
            task=task,
            process_id=process_id,
            task_index=task_index,
            solution_config=solution_config,
            online_controller=online_controller,
            gpu_id = gpu_id
        )
        output_queue.put((task_id, metrics, explore_time, all_time))

    if args.use_offline_ai2thor:
        online_controller.stop()
        controller.stop()
    else:
        controller.stop()


def preparation():
    '''
      Experimental settings: different solutions options (choose one setting for each module)
      {
        'seg_utils': ['GT', 'MaskRCNN', 'yolov7']
        'walkthrough_search': ['cover','cover_nearest', 'cover_continue','minViewDistance','mass']
        'unshuffle_search': ['cover', 'cover_nearest','same', 'localAuto','mass']
        'unshuffle_match':  ['feature_IoU_based', 'feature_centroid_based', 'feature_relation_based']
        'unshuffle_reorder': ['greedy', 'or_tools', 'random']
      }
    '''
    solution_config = {
        'seg_utils': args.seg_utils,
        'walkthrough_search': args.walkthrough_search,
        'unshuffle_search': args.unshuffle_search,
        'unshuffle_match': args.unshuffle_match,
        'unshuffle_reorder': args.unshuffle_reorder
    }

    assert args.seg_utils in ['GT', 'MaskRCNN', 'yolov7']
    assert args.walkthrough_search in ['cover', 'cover_nearest','cover_continue', 'minViewDistance','mass']
    assert args.unshuffle_search in ['cover','cover_continue','cover_nearest','same', 'localAuto','minViewDistance','mass']
    assert args.unshuffle_match in ['feature_IoU_based', 'feature_centroid_based', 'feature_relation_based','feature_pointcloud_based'] 
    assert args.unshuffle_reorder in ['greedy', 'or_tools', 'random']
    return solution_config


if __name__ == '__main__':
    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    ptitle("Rearrange Tasks")
    ssl._create_default_https_context = ssl._create_unverified_context
    nprocesses = (
        max((2 * mp.cpu_count()) // 4, 1) if platform.system() == "Linux" else 1
    )
    nprocesses = 1
    torch.multiprocessing.set_start_method('spawn')

    print('Nprocesses: ', nprocesses)
    available_devices_num = 3
    solution_config = preparation()
    print("Solution config: ", solution_config)

    if not os.path.exists(args.store_explorer_path):
        os.mkdir(args.store_explorer_path)

    # load task dataset
    split = args.split.lower()

    task_id_to_tasks = load_dataset_from_path(split)

    split = 'train' if split == 'debug' else split    # only used for debug
    

    task_to_metrics = {}
    assert args.test_mode in ['two_phase','only_walkthrough','only_walkthrough_steps']
    if args.test_mode == 'only_walkthrough' or args.test_mode == 'only_walkthrough_steps':
        metric_file = f"./metrics/metrics_{args.seg_utils}{args.test_mode}|{args.walkthrough_search}|{args.only_walkthrough_mode_max_steps}.json"
    else:
        metric_file = f"./metrics/metrics_{args.seg_utils}{args.test_mode}|{args.walkthrough_search}|{args.unshuffle_search}|{args.unshuffle_match}|{args.unshuffle_reorder}.json"
    if not os.path.exists(os.path.dirname(metric_file)):
        os.mkdir(os.path.dirname(metric_file))
    
    args.movie_dir += f"{args.seg_utils}{args.test_mode}|{args.walkthrough_search}|{args.unshuffle_search}|{args.unshuffle_match}|{args.unshuffle_reorder}"

    runtime_file = f'./metrics/runtime_{args.seg_utils}{args.test_mode}|{args.walkthrough_search}|{args.unshuffle_search}|{args.unshuffle_match}|{args.unshuffle_reorder}.json'
    runtime_record = dict()

    if args.load_submission:
        if os.path.exists(metric_file):
            with open(metric_file, 'rt', encoding='UTF-8') as f:
                task_to_metrics = json.load(f)
        
        if os.path.exists(runtime_file):
            with open(runtime_file, "r") as f:
                runtime_record = json.load(f)



    num_tasks_to_run = 0
    send_queue = mp.Queue()

    # room_list = ["FloorPlan10_"+str(i) for i in range(50) if i%2 == 0]
    for task_id,task in task_id_to_tasks.items():
            if task_id in ["FloorPlan10_36"]:
            # if task_id in ["FloorPlan28_0"]:
            # if task_id in ['FloorPlan28_1', 'FloorPlan28_3', 'FloorPlan28_5', 'FloorPlan28_7', 'FloorPlan28_9', 'FloorPlan26_1', 'FloorPlan26_3', 'FloorPlan26_5', 'FloorPlan26_7', 'FloorPlan26_9', 'FloorPlan30_1', 'FloorPlan30_3', 'FloorPlan30_5', 'FloorPlan30_7', 'FloorPlan30_9', 'FloorPlan27_1', 'FloorPlan27_3', 'FloorPlan27_5', 'FloorPlan27_7', 'FloorPlan27_9', 'FloorPlan226_1', 'FloorPlan226_3', 'FloorPlan226_5', 'FloorPlan226_7', 'FloorPlan226_9', 'FloorPlan228_1', 'FloorPlan228_3', 'FloorPlan228_5', 'FloorPlan228_7', 'FloorPlan228_9', 'FloorPlan330_1', 'FloorPlan330_3', 'FloorPlan330_5', 'FloorPlan330_7', 'FloorPlan330_9', 'FloorPlan328_1', 'FloorPlan328_3', 'FloorPlan328_5', 'FloorPlan328_7', 'FloorPlan328_9', 'FloorPlan329_1', 'FloorPlan329_3', 'FloorPlan329_5', 'FloorPlan329_7', 'FloorPlan329_9', 'FloorPlan426_1', 'FloorPlan426_3', 'FloorPlan426_5', 'FloorPlan426_7', 'FloorPlan426_9', 'FloorPlan327_1', 'FloorPlan327_3', 'FloorPlan327_5', 'FloorPlan327_7', 'FloorPlan327_9', 'FloorPlan229_1', 'FloorPlan229_3', 'FloorPlan229_5', 'FloorPlan229_7', 'FloorPlan229_9', 'FloorPlan29_1', 'FloorPlan29_3', 'FloorPlan29_5', 'FloorPlan29_7', 'FloorPlan29_9', 'FloorPlan429_1', 'FloorPlan429_3', 'FloorPlan429_5', 'FloorPlan429_7', 'FloorPlan429_9', 'FloorPlan227_1', 'FloorPlan227_3', 'FloorPlan227_5', 'FloorPlan227_7', 'FloorPlan227_9', 'FloorPlan326_1', 'FloorPlan326_3', 'FloorPlan326_5', 'FloorPlan326_7', 'FloorPlan326_9', 'FloorPlan428_1', 'FloorPlan428_3', 'FloorPlan428_5', 'FloorPlan428_7', 'FloorPlan428_9', 'FloorPlan427_1', 'FloorPlan427_3', 'FloorPlan427_5', 'FloorPlan427_7', 'FloorPlan427_9', 'FloorPlan430_1', 'FloorPlan430_3', 'FloorPlan430_5', 'FloorPlan430_7', 'FloorPlan430_9']:
            # if task_id not in task_to_metrics:
                send_queue.put((task_id,task))
                num_tasks_to_run += 1
            else:
                print(
                    f"{task_id} in submission already.. skipping..")

    print('Still need to test num: ', num_tasks_to_run)
    receive_queue = mp.Queue()
    process = []
    for i in range(nprocesses):
        # gpu_id = i % 3 + 1
        gpu_id = i % 2
        p = mp.Process(
            target=rearrange_worker,
            kwargs=dict(
                input_queue=send_queue,
                output_queue=receive_queue,
                process_id=i,
                solution_config=solution_config,
                x_display=f"2.1",
                gpu_id = gpu_id,

            )
        )
        p.start()
        process.append(p)
        time.sleep(0.5)

    num_received = 0
    while num_tasks_to_run > num_received:
        try:
            task_id, metrics, explore_time, all_time = receive_queue.get(timeout=1)
            num_received += 1
        except queue.Empty:
            continue

        task_to_metrics[task_id] = metrics
        runtime_record[task_id] = [explore_time, all_time]
        print(f'Received: {task_id}')

        with open(metric_file, 'w') as f:
            json.dump(task_to_metrics, f, indent=4, cls=NumpyJSONEncoder)
            print(f"~~~~~~~~~~~~~~Saving {task_id} to {metric_file}")

        with open(runtime_file, "w") as f:
            json.dump(runtime_record, f, indent=4, cls=NumpyJSONEncoder)

    for p in process:
        try:
            p.join(timeout=1)
        except:
            pass
