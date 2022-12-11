import os
from pathlib import Path

import ffmpeg
def encode_video(framerate = 10)->Path:
    '''Encodes png animation frames in ./animation to a mp4 file'''
    os.chdir(str(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))))
    pnglist = list(Path("./animation/").rglob("*.png"))
    mp4list = list(Path("./animation/").rglob("*.mp4"))
    if mp4list == None:
        mp4list[0] = 1
    lownum = 100000
    for file in pnglist:
        try:
            if int(file.stem)<lownum:
                lownum = int(file.stem)
        except:
            pass
    (
        ffmpeg.input(
            ('./animation/%d.png'),pattern_type='sequence',start_number=lownum,framerate=framerate,thread_queue_size = 2048)
            .output("./animation/" + str(len(mp4list)) + ".mp4",preset='slower')
            .run()
    )
    return Path("./animation/" + str(len(mp4list)) + ".mp4")
#encode_video()