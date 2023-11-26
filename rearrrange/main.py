import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from rearrange_on_proc.rearrange_challenge.constants import OTHER_INFO_DIR
from rearrange_on_proc.arguments import args
from rearrange_on_proc.tasks import RearrangementTask
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
 



def load_dataset_from_path(split, areaRanges, roomNum):
    data_path = os.path.abspath(os.path.join(DATASET_DIR, f"{split}_0424.pkl.gz"))
    # data_path = "/home/lyy/rearrange_on_ai2thor/rearrange_on_proc/data/2022/test.pkl.gz"
    # data_path = os.path.abspath(os.path.join(
    #     DATASET_DIR, f"debug_add33.pkl.gz"))
    if not os.path.exists(data_path):
        raise RuntimeError(f"No data at path {data_path}")

    print(f'Loading {split} dataset...')
    data = compress_pickle.load(path=data_path)

    if roomNum != 'all':
        # with open(os.path.join(OTHER_INFO_DIR, 'all_rooms/statistic.json'), "r") as f:
        #     statistic = json.load(f)
        # split_statistic = statistic[split]
        with open(os.path.join(OTHER_INFO_DIR, 'all_rooms/dataset_test_val_train_multi_instance.json'), "r") as f:
            statistic = json.load(f)
        split_statistic = statistic[split]

        candidate_scenes = []
        for num_str in roomNum:
            for scene in split_statistic[num_str]:
                if scene in data:
                    candidate_scenes.append(scene)
        print(
            f'Loading {roomNum} scenes (with scenes nums: {len(candidate_scenes)})')

        choose_data = itemgetter(*candidate_scenes)(data)

    else:
        print(f'Loading all scenes (with scenes nums: {len(data)})')
        candidate_scenes = list(data.keys())
        choose_data = itemgetter(*candidate_scenes)(data)
        
    return choose_data


def run_rearrange_task(controller, task, task_areaRange, all_dataset, process_id, task_index, solution_config, online_controller):
    aithor_rearrangement = RearrangementTask(controller, task, task_areaRange, all_dataset, process_id, task_index, solution_config, online_controller)
    task_id, metrics, explore_time, all_time  = aithor_rearrangement.main()
    return task_id, metrics, explore_time, all_time

def rearrange_worker(
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    process_id,
    all_dataset,
    solution_config,
    x_display
):
    task_index = 0
    # device = torch.device(f"cuda:{process_id % available_devices_num}") if available_devices_num > 0 and torch.cuda.is_available(
    # ) else torch.device('cpu')
    # x_display_id = process_id % available_devices_num if available_devices_num > 0  else 0

    online_controller = Controller(scene=all_dataset['train'][0],
                            x_display=x_display,
                            width=args.W,
                            height=args.H,
                            fieldOfView=args.fov,
                            gridSize=args.STEP_SIZE,
                            rotateStepDegrees=args.DT,
                            renderDepthImage=True,
                            renderInstanceSegmentation=True,
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
            task, areaRange = input_queue.get(timeout=2)
            task_index += 1
        except queue.Empty:
            break

        task_id, metrics, explore_time, all_time = run_rearrange_task(
            controller=controller,
            task=task,
            task_areaRange=areaRange,
            all_dataset=all_dataset,
            process_id=process_id,
            task_index=task_index,
            solution_config=solution_config,
            online_controller=online_controller
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
    nprocesses = 30
    torch.multiprocessing.set_start_method('spawn')

    print('Nprocesses: ', nprocesses)
    available_devices_num = 3
    dataset = prior.load_dataset("procthor-10k")
    solution_config = preparation()
    print("Solution config: ", solution_config)

    if not os.path.exists(args.store_explorer_path):
        os.mkdir(args.store_explorer_path)

    # load task dataset
    split = args.split.lower()
    areaRanges = None
    # if args.areaRanges == 'all':
    #     areaRanges = 'all'
    # else:
    #     areaRanges = args.areaRanges.split('|')
    roomNum = None
    if args.roomNum == 'all':
        roomNum = 'all'
    else:
        roomNum = [str(x)+'rooms' for x in args.roomNum.split('|')]
    print(f'Select areaRanges: {areaRanges}, roomNum: {roomNum}')

    scene_to_tasks = load_dataset_from_path(split=split, areaRanges=areaRanges, roomNum=roomNum)
    # load area statistic.json
    with open(os.path.join(OTHER_INFO_DIR, 'all_rooms/statistic.json'), "r") as f:
        statistic = json.load(f)

    split = 'train' if split == 'debug' else split    # only used for debug
    split_statistic = statistic[split] if split in statistic.keys() else {}
    scene_to_areaRange = {}
    for area, scenes in split_statistic.items():
        for scene in scenes:
            scene_to_areaRange[scene] = area

    task_to_metrics = {}
    assert args.test_mode in ['two_phase','only_walkthrough','only_walkthrough_steps']
    if args.test_mode == 'only_walkthrough' or args.test_mode == 'only_walkthrough_steps':
        metric_file = f"./metrics/metrics_{str(roomNum)}_{args.seg_utils}{args.test_mode}|{args.walkthrough_search}|{args.only_walkthrough_mode_max_steps}.json"
    else:
        metric_file = f"./metrics/metrics_{str(roomNum)}_{args.seg_utils}{args.test_mode}|{args.walkthrough_search}|{args.unshuffle_search}|{args.unshuffle_match}|{args.unshuffle_reorder}.json"
    if not os.path.exists(os.path.dirname(metric_file)):
        os.mkdir(os.path.dirname(metric_file))
    
    args.movie_dir += f"_{str(roomNum)}_{args.seg_utils}{args.test_mode}|{args.walkthrough_search}|{args.unshuffle_search}|{args.unshuffle_match}|{args.unshuffle_reorder}"

    runtime_file = f'./metrics/runtime_{str(roomNum)}.json'
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

    tasks_json_tuple = tuple(scene_to_tasks)
    for tasks in tasks_json_tuple:
        for task in tasks:
            if task['unique_id'] not in task_to_metrics:
            # if task['unique_id'] in ['train_5315_2']:
            # if task['unique_id'] in ['test_75_2']:
            # if task['unique_id'] in ['test_629_1', 'test_629_2', 'test_402_1', 'test_402_2', 'test_486_1']:
                scene_str = "_".join(task['unique_id'].split("_")[:-1])
                if 'debug' in scene_str:
                    scene_str = scene_str.replace(
                        "debug", 'train')  # only used for debug
                send_queue.put((task, scene_to_areaRange[scene_str]))
                num_tasks_to_run += 1
            else:
                print(
                    f"{task['unique_id']} in submission already.. skipping..")

    print('Still need to test num: ', num_tasks_to_run)
    receive_queue = mp.Queue()
    process = []
    for i in range(nprocesses):
        # gpu_id = i % 3 + 1
        gpu_id = 1
        p = mp.Process(
            target=rearrange_worker,
            kwargs=dict(
                input_queue=send_queue,
                output_queue=receive_queue,
                process_id=i,
                all_dataset=dataset,
                solution_config=solution_config,
                x_display=f"2.{gpu_id}"

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
