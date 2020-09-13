import av
import jpeg4py as jpeg
import os

def av_decode_video(video_path):
    with av.open(video_path) as container:
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_rgb().to_ndarray())
    return frames

def av_read_frames(video_path, frames_num = 0, is_right = True):
    if frames_num > 25:
        frames_num = 25
    image_paths = []
    if is_right:
        for cur_t in range(frames_num):
            image_paths.append(os.path.join(video_path,"{}.jpg".format(video_path,cur_t)))
    else:
        for cur_t in range(frames_num):
            image_paths.append(os.path.join(video_path,"{}.jpg".format(24-cur_t)))
    imgs = [jpeg.JPEG(image_path).decode() for image_path in image_paths]
    return imgs

def ava_read_frames(video_path, cur_frames, frame_num=0, is_right = True):
    image_paths = []
    video_name = video_path.split('/')[-1]
    if is_right:
        for cur_t in range(frame_num):
            image_paths.append(os.path.join(video_path,"{0}_{1:0>6d}.jpg".format(video_name,cur_t+cur_frames)))
    else:
        for cur_t in range(frame_num):
            image_paths.append(os.path.join(video_path,"{0}_{1:0>6d}.jpg".format(video_name,cur_frames-frame_num+cur_t)))
    imgs = [jpeg.JPEG(image_path).decode() for image_path in image_paths]
    #print(len(imgs))
    return imgs