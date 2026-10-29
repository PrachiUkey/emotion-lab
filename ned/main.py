"""Emotion-edit a talking-head video with NED.

    python main.py --video in.mp4 --emotion happy --out out_happy.mp4
    python main.py --video in.mp4 --emotion happy angry --out out.mp4      # first half happy, second half angry
    python main.py --video tarantino.mp4 --emotion sad --renderer_ckpt ./checkpoints_tarantino --out out.mp4

The face renderer is person-specific. Without --renderer_ckpt, the meta-renderer is
fine-tuned on the input video itself (slow, hours) and cached under
<work_dir>/<person>/renderer, so later runs for the same person reuse it.

Input requirements: one person, face visible in every frame (trim intros/title cards/cutaways first;
eye_landmarks.py crashes if the video starts without a face). A few minutes of footage gives a better
renderer for a new person.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import cv2

REPO = os.path.dirname(os.path.abspath(__file__))
EMOTIONS = ['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised']
META_RENDERER = os.path.join(REPO, 'checkpoints_meta-renderer')
MANIPULATOR = os.path.join(REPO, 'manipulator_checkpoints')


def run(step, cmd):
    print(f'\n===== [{step}] {" ".join(cmd)}', flush=True)
    t = time.time()
    # scripts import sibling packages (e.g. `from preprocessing...`), so the repo root must be importable
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [REPO, os.environ.get('PYTHONPATH')])))
    res = subprocess.run(cmd, cwd=REPO, env=env)
    if res.returncode != 0:
        sys.exit(f'\n!!!!! FAILED at step "{step}" (exit code {res.returncode}). Aborting.')
    print(f'===== [{step}] done in {time.time() - t:.0f}s', flush=True)


def stage_video(video, celeb_dir):
    """Place the input video as <celeb_dir>/videos/<name>.mp4 (the layout NED's scripts expect)."""
    videos_dir = os.path.join(celeb_dir, 'videos')
    os.makedirs(videos_dir, exist_ok=True)
    dst = os.path.join(videos_dir, os.path.basename(video))
    if not os.path.exists(dst):
        shutil.copy2(video, dst)
    return dst


def preprocess(celeb_dir, mode, gpu):
    py = sys.executable
    g = ['--gpu_id', gpu]
    if mode == 'train':
        run('preprocess/detect', [py, 'preprocessing/detect.py', '--celeb', celeb_dir, '--split', *g])
        run('preprocess/eye_landmarks', [py, 'preprocessing/eye_landmarks.py', '--celeb', celeb_dir, '--mouth', '--align', *g])
    else:
        run('preprocess/detect', [py, 'preprocessing/detect.py', '--celeb', celeb_dir, '--save_videos_info', '--save_full_frames', *g])
        run('preprocess/eye_landmarks', [py, 'preprocessing/eye_landmarks.py', '--celeb', celeb_dir, '--align', *g])
    run('preprocess/segment_face', [py, 'preprocessing/segment_face.py', '--celeb', celeb_dir, *g])
    run('preprocess/reconstruct', [py, 'preprocessing/reconstruct.py', '--celeb', celeb_dir, '--save_shapes', '--save_nmfcs', *g])
    run('preprocess/align', [py, 'preprocessing/align.py', '--celeb', celeb_dir,
                             '--faces_and_masks', '--shapes', '--nmfcs', '--landmarks'])


def train_renderer(train_dir, renderer_dir, epochs, gpu):
    py = sys.executable
    cmd = [py, 'renderer/train.py', '--celeb', train_dir, '--checkpoints_dir', renderer_dir,
           '--niter', str(epochs), '--gpu_ids', gpu]
    if os.path.exists(os.path.join(renderer_dir, 'latest_net_G.pth')):
        cmd += ['--continue_train', '--which_epoch', 'latest']
    else:
        cmd += ['--load_pretrain', META_RENDERER, '--which_epoch', '15']
    run('renderer/train', cmd)


def main():
    parser = argparse.ArgumentParser(description='Change the emotion of the person in a video (NED).')
    parser.add_argument('--video', required=True, help='Input video (single visible face).')
    parser.add_argument('--emotion', required=True, nargs='+', choices=EMOTIONS,
                        help='Target emotion(s). Several = video split into equal consecutive segments.')
    parser.add_argument('--out', required=True, help='Output .mp4 path.')
    parser.add_argument('--person', default=None, help='Identity name for caching (default: video file stem).')
    parser.add_argument('--renderer_ckpt', default=None,
                        help='Existing renderer checkpoint dir (e.g. ./checkpoints_tarantino). Skips training.')
    parser.add_argument('--renderer_epochs', type=int, default=20, help='Fine-tuning epochs for a new person.')
    parser.add_argument('--work_dir', default=os.path.join(REPO, 'runs'))
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    video = os.path.abspath(args.video)
    if not os.path.isfile(video):
        sys.exit(f'Input video not found: {video}')
    out = os.path.abspath(args.out)
    person = args.person or os.path.splitext(os.path.basename(video))[0]
    person_dir = os.path.join(os.path.abspath(args.work_dir), person)
    test_dir = os.path.join(person_dir, 'test')
    exp_name = '_'.join(args.emotion)
    py = sys.executable
    t0 = time.time()

    # 1. Renderer for this identity
    if args.renderer_ckpt:
        renderer_dir, which_epoch = os.path.abspath(args.renderer_ckpt), '20'
    else:
        renderer_dir, which_epoch = os.path.join(person_dir, 'renderer'), str(args.renderer_epochs)
        if os.path.exists(os.path.join(renderer_dir, f'{which_epoch}_net_G.pth')):
            print(f'Using cached renderer for "{person}": {renderer_dir}')
        else:
            train_dir = os.path.join(person_dir, 'train')
            stage_video(video, train_dir)
            preprocess(train_dir, 'train', args.gpu)
            train_renderer(train_dir, renderer_dir, args.renderer_epochs, args.gpu)
    if not os.path.exists(os.path.join(renderer_dir, f'{which_epoch}_net_G.pth')):
        sys.exit(f'Renderer checkpoint {which_epoch}_net_G.pth not found in {renderer_dir}')

    # 2. Preprocess the video to edit
    staged = stage_video(video, test_dir)
    preprocess(test_dir, 'test', args.gpu)

    # 3. Manipulate expressions, render, unalign, blend
    run('manipulate', [py, 'manipulator/test.py', '--celeb', test_dir, '--checkpoints_dir', MANIPULATOR,
                       '--trg_emotions', *args.emotion, '--exp_name', exp_name, '--gpu_ids', args.gpu])
    run('render/create_inputs', [py, 'renderer/create_inputs.py', '--celeb', test_dir, '--exp_name', exp_name,
                                 '--save_shapes', '--gpu_id', args.gpu])
    run('render/test', [py, 'renderer/test.py', '--celeb', test_dir, '--exp_name', exp_name,
                        '--checkpoints_dir', renderer_dir, '--which_epoch', which_epoch, '--gpu_ids', args.gpu])
    run('unalign', [py, 'postprocessing/unalign.py', '--celeb', test_dir, '--exp_name', exp_name])
    run('blend', [py, 'postprocessing/blend.py', '--celeb', test_dir, '--exp_name', exp_name,
                  '--save_images', '--gpu_id', args.gpu])

    # 4. Frames -> video at the source fps, with the original audio.
    # The temporal renderer emits nothing for the first frames (no history yet); fill those with the
    # originals so the frame count matches the source and the audio stays in sync.
    orig_frames = os.path.join(test_dir, 'full_frames')
    out_frames = os.path.join(test_dir, exp_name, 'full_frames')
    missing = sorted(set(os.listdir(orig_frames)) - set(os.listdir(out_frames)))
    if missing:
        print(f'Filling {len(missing)} frame(s) without a rendered output with the original: {missing[:5]}...')
        for name in missing:
            shutil.copy2(os.path.join(orig_frames, name), os.path.join(out_frames, name))

    cap = cv2.VideoCapture(staged)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0:
        sys.exit(f'Could not read fps from {staged}')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    run('images2video', [py, 'postprocessing/images2video.py',
                         '--imgs_path', out_frames,
                         '--out_path', out, '--fps', str(fps), '--audio', staged])
    if not os.path.isfile(out):
        sys.exit(f'Pipeline finished but no output file at {out}')
    print(f'\nDone in {(time.time() - t0) / 60:.1f} min -> {out}')


if __name__ == '__main__':
    main()
