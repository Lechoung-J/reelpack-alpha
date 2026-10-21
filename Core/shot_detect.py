from frame_sampling import shot_frame_positions
"""Content-aware visual boundaries; confidence is heuristic, not calibrated probability."""
from pathlib import Path
import hashlib
import json
import math
import subprocess
import os


def normalize_video(video_path, cache_dir):
    """Never modify source. Cache an atomic, video-only H264 proxy at original rate."""
    source = Path(video_path).resolve()
    stat = source.stat()
    key = hashlib.sha256(f'{source}:{stat.st_size}:{stat.st_mtime_ns}:h264-v1'.encode()).hexdigest()[:24]
    folder = Path(cache_dir); folder.mkdir(parents=True, exist_ok=True)
    target = folder / f'{key}.mp4'
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                            '-show_entries', 'stream=codec_name', '-of', 'json', str(source)],
                           capture_output=True, text=True, check=True, timeout=30)
    codec = json.loads(probe.stdout)['streams'][0]['codec_name']
    if not target.exists():
        temp = folder / f'{key}.{os.getpid()}.tmp.mp4'
        try:
            subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(source), '-map', '0:v:0',
                            '-an', '-vf', 'scale=640:-2', '-c:v', 'libx264', '-preset', 'veryfast',
                            '-crf', '18', '-pix_fmt', 'yuv420p', str(temp)], check=True, timeout=1800)
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
    return target, {'codec': 'h264', 'source_codec': codec, 'cache_key': key, 'proxy_width': 640}


class FFmpegCapture:
    """Bounded-memory raw decoder fallback for hosts with broken OpenCV video I/O."""
    def __init__(self, path):
        import cv2
        self.path = str(path); self.process = None; self.position = 0
        probe = subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate,nb_frames','-of','json',self.path],capture_output=True,text=True,check=True,timeout=30)
        stream = json.loads(probe.stdout)['streams'][0]
        self.width = stream['width']; self.height = stream['height']
        a,b = stream['avg_frame_rate'].split('/'); self.fps = float(a)/float(b)
        self.total = int(stream['nb_frames']); self.set(cv2.CAP_PROP_POS_FRAMES,0)
    def isOpened(self): return True
    def get(self, prop):
        import cv2
        return self.fps if prop == cv2.CAP_PROP_FPS else self.total
    def set(self, prop, position):
        position = int(position)
        if self.process is not None and position == self.position: return
        self.release(); self.position = position
        self.process = subprocess.Popen(['ffmpeg','-v','error','-ss',str(position/self.fps),'-i',self.path,'-f','rawvideo','-pix_fmt','bgr24','-threads','1','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    def read(self):
        import numpy as np
        size = self.width*self.height*3
        data = self.process.stdout.read(size)
        if len(data) != size: return False, None
        self.position += 1
        return True,np.frombuffer(data,dtype=np.uint8).reshape(self.height,self.width,3)
    def release(self):
        if self.process is not None:
            self.process.stdout.close(); self.process.terminate(); self.process.wait(); self.process = None


def activity_mask(samples):
    import numpy as np
    import cv2
    data = np.asarray(samples, dtype=np.float32)
    # Per-pixel temporal spread, then block aggregation prevents isolated compression noise.
    spread = np.std(data, axis=0).mean(axis=2)
    blocks = cv2.resize(spread, (16, 9), interpolation=cv2.INTER_AREA)
    cutoff = max(2.5, float(np.percentile(blocks, 80)) * .10)
    active = blocks > cutoff
    fallback = bool(active.sum() < 8)
    if fallback:
        active[:] = True
    mask = cv2.resize(active.astype('uint8'), (data.shape[2], data.shape[1]), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(mask)
    region = {'x': float(xs.min()/mask.shape[1]), 'y': float(ys.min()/mask.shape[0]),
              'width': float((xs.max()-xs.min()+1)/mask.shape[1]), 'height': float((ys.max()-ys.min()+1)/mask.shape[0])}
    static = [{'x': x/16, 'y': y/9, 'width': 1/16, 'height': 1/9}
              for y in range(9) for x in range(16) if not active[y, x]]
    return mask, region, static, fallback


def select_boundaries(features, fps):
    import numpy as np
    f = np.asarray(features)
    if not len(f): return [], {}
    scores = .50*f[:, 0] + .30*f[:, 1] + .20*f[:, 2]
    # Robust global baseline excludes the strongest transitions from noise estimation.
    base = scores[scores <= np.percentile(scores, 80)]
    floor = max(.045, float(base.mean()+3*base.std()))
    candidates = []
    radius = max(2, round(fps*.6))
    for i, score in enumerate(scores):
        if i == 0: continue
        local = scores[max(1, i-radius):min(len(scores), i+radius+1)]
        quiet = local[local <= np.percentile(local, 70)]
        limit = max(floor, float(quiet.mean()+3*quiet.std()))
        if score < limit or score < max(scores[max(1,i-2):i+3]): continue
        # Require actual pixel change: ECR alone reacts strongly to talking-head motion.
        if f[i, 2] < .035 or (f[i, 0] < .055 and f[i, 2] < .14): continue
        confidence = float(min(.99, .5 + .25*(score/limit-1)))
        candidates.append((i, float(score), confidence))
    selected = []
    for candidate in sorted(candidates, key=lambda c: c[1], reverse=True):
        if all(abs(candidate[0]-c[0]) >= max(2, round(fps*.16)) for c in selected): selected.append(candidate)
    return sorted(selected), {'global_threshold': floor, 'k': 3, 'minimum_shot_seconds': .16}


def detect_shots_custom(video_path, package_dir, threshold=0.28, representative_interval=2.0, cache_dir=None):
    if not 0 < threshold <= 1 or representative_interval <= 0: raise ValueError('invalid shot detector parameters')
    try:
        import cv2
        import numpy as np
    except ImportError:
        return {'status': 'not_run', 'reason': 'cv2/numpy unavailable', 'items': []}, []
    cap = None
    try:
        path, normalization = normalize_video(video_path, cache_dir or Path(package_dir)/'_decode_cache')
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
            cap.release(); cap = FFmpegCapture(path)
            normalization['decoder'] = 'ffmpeg-raw-fallback'
            fps = cap.fps; total = cap.total
        samples = []
        for pos in np.linspace(0, max(0,total-1), min(96,total)).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos)); ok, image = cap.read()
            if ok: samples.append(cv2.resize(image, (160,90)))
        if not samples: raise RuntimeError('no decoded frames')
        mask, region, static, fallback = activity_mask(samples)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        previous = None; features = []; kernel = np.ones((3,3),np.uint8)
        while True:
            ok, image = cap.read()
            if not ok: break
            small = cv2.resize(image,(160,90)); hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv],[0,1,2],mask,[16,8,8],[0,180,0,256,0,256]); hist /= max(hist.sum(),1)
            edges = cv2.Canny(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),70,140)*mask
            dilated = cv2.dilate(edges,kernel)
            if previous is None: features.append((0.,0.,0.))
            else:
                old, oldhist, oldedges, olddilated = previous
                histogram = cv2.compareHist(hist,oldhist,cv2.HISTCMP_CHISQR_ALT)/4
                ecr = max(float(((edges>0)&(olddilated==0)).sum()/max((edges>0).sum(),1)),
                          float(((oldedges>0)&(dilated==0)).sum()/max((oldedges>0).sum(),1)))
                pixel = float(np.abs(small.astype(float)-old).mean(axis=2)[mask>0].mean()/255)
                features.append((histogram,ecr,pixel))
            previous = (small.astype(float),hist,edges,dilated)
        cuts, parameters = select_boundaries(features,fps); count = len(features)
        cuts = [(0,None,1.)]+cuts; evidence=[]; items=[]
        folder=Path(package_dir)/'evidence'; folder.mkdir(parents=True,exist_ok=True)
        for index,(first,score,confidence) in enumerate(cuts):
            last=cuts[index+1][0] if index+1<len(cuts) else count
            sid=f'shot_{index+1:04d}'; refs=[]
            positions=shot_frame_positions(first, last, fps, representative_interval)
            for number,pos in enumerate(positions):
                cap.set(cv2.CAP_PROP_POS_FRAMES,pos); ok,image=cap.read()
                if not ok: continue
                eid=f'{sid}_frame_{number:04d}'; relative=f'evidence/{eid}.jpg'
                if cv2.imwrite(str(Path(package_dir)/relative),image):
                    refs.append(eid); evidence.append({'evidence_id':eid,'kind':'image','timestamp':pos/fps,'relative_path':relative,'shot_id':sid,'purpose':'shot_representative'})
            items.append({'shot_id':sid,'start':first/fps,'end':last/fps,'boundary':{'score':score,'confidence':confidence,'uncertain':fallback or (index>0 and confidence<.75),'resolution_seconds':1/fps,'error_seconds':None,'type':'media_start' if index==0 else 'visual_cut_candidate'},'representative_frame':refs[len(refs)//2] if refs else None,'evidence_ids':refs})
        return {'status':'complete' if count == total and all(i['evidence_ids'] for i in items) else 'partial','backend':'opencv-hsv-ecr-adaptive-v1','semantic_scenes':'not_run','threshold':threshold,'threshold_mode':'adaptive (legacy threshold accepted, not used)','parameters':parameters,'decoded_frames':count,'expected_frames':total,'sampling_fps':fps,'normalization':normalization,'active_region':region,'static_regions':static,'activity_fallback':fallback,'representative_interval_seconds':min(2.0, representative_interval),'items':items}, evidence
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        return {'status':'failed','reason':str(exc),'items':[]}, []
    finally:
        if cap is not None: cap.release()


def detect_shots(video_path, package_dir, threshold=None, representative_interval=2.0,
                 backend=None, fallback=True, cache_dir=None, model_dir=None, threads=4):
    """Prefer TransNetV2; explicit fallback is recorded, never reported as network output.

    REELPACK_SHOT_BACKEND=custom restores the previous implementation.
    For strict evaluation use backend=transnetv2, fallback=False.
    """
    selected = backend or os.environ.get('REELPACK_SHOT_BACKEND', 'transnetv2')
    if selected not in {'custom', 'transnetv2'}:
        raise ValueError(f'unknown shot backend: {selected}')
    if selected == 'custom':
        return detect_shots_custom(video_path, package_dir, .28 if threshold is None else threshold,
                                   representative_interval, cache_dir=cache_dir)
    python = os.environ.get('REELPACK_TRANSNET_PYTHON',
                            str(Path(__file__).resolve().parent.parent / '.venv-transnet/bin/python'))
    result, evidence = detect_shots_isolated(python, video_path, package_dir,
        .5 if threshold is None else threshold, representative_interval, cache_dir, model_dir, threads)
    result['requested_backend'] = selected
    result['fallback'] = None
    if fallback and result['status'] in {'not_run', 'failed'}:
        failure = {key: result[key] for key in ('backend', 'status', 'reason')}
        result, evidence = detect_shots_custom(video_path, package_dir, .28, representative_interval, cache_dir=cache_dir)
        result['requested_backend'] = selected
        result['fallback'] = failure
    return result, evidence


def detect_shots_isolated(python, video_path, package_dir, threshold=.5,
                          representative_interval=2., cache_dir=None, model_dir=None, threads=4):
    """Explicit environment isolation; errors remain eligible for truthful fallback."""
    import tempfile
    config = dict(video_path=str(video_path), package_dir=str(package_dir), threshold=threshold,
                  representative_interval=representative_interval, threads=threads,
                  cache_dir=str(cache_dir) if cache_dir else None,
                  model_dir=str(model_dir) if model_dir else None)
    try:
        with tempfile.TemporaryDirectory() as td:
            destination = Path(td) / 'result.json'
            run = subprocess.run([str(python), str(Path(__file__).with_name('shot_detect_transnet.py')),
                                  json.dumps(config), str(destination)],
                                 capture_output=True, text=True, timeout=1800)
            if run.returncode != 0 or not destination.exists():
                raise RuntimeError(run.stderr[-2000:] or f'TransNet worker exit {run.returncode}')
            payload = json.loads(destination.read_text())
            return payload['shots'], payload['evidence']
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        return {'status': 'failed', 'backend': 'transnetv2-pytorch-cpu', 'reason': str(exc), 'items': []}, []
