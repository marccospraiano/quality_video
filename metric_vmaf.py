#!/usr/bin/env python
# Copyright (c) 2013 The WebRTC project authors. All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the LICENSE file in the root of the source
# tree. An additional intellectual property rights grant can be found
# in the file PATENTS.  All contributing project authors may
# be found in the AUTHORS file in the root of the source tree.

import json
import optparse
import os
import shutil
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Chrome browsertests will throw away stderr; avoid that output gets lost.
sys.stderr = sys.stdout

#ref_video = 'ref_video'
#test_video = 'test_video'
#frame_analyzer = 'frame_analyzer'
#aligned_output_file = 'aligned_output_file'
#vmaf_model = 'vmaf'                                # Path to VMAF model
#yuv_frame_width = '640'
#yuv_frame_height = '360'
#chartjson_result_file = 'results_vmaf'
#vmaf = 'vmaf'


def _DevNull():
    
    """On Windows, sometimes the inherited stdin handle from the parent process
    fails. Workaround this by passing null to stdin to the subprocesses commands.
    This function can be used to create the null file handler.
    """
    return open(os.devnull, 'r')


def _RunVmaf(yuv_directory, logfile):
    """ Run VMAF to compare videos and print output.
    The yuv_directory is assumed to have been populated with a reference and test
    video in .yuv format, with names according to the label.
    """
    yuv_frame_width = 640
    yuv_frame_height = 360
    vmaf_model = 'vmaf'
    
    cmd = [
        'yuv420p',
        str(yuv_frame_width),
        str(yuv_frame_height),
        os.path.join(yuv_directory, "videoSRC001_640x360_30_qp_00.yuv"),
        os.path.join(yuv_directory, "videoSRC001_640x360_30_qp_23.yuv"),
        vmaf_model,
        '--log',
        logfile,
        '--log-fmt',
        'json',
    ]
    
    vmaf = subprocess.Popen(cmd, stdin=_DevNull(),
                            stdout=sys.stdout, stderr=sys.stderr)
    vmaf.wait()
    if vmaf.returncode != 0:
        print('Failed to run VMAF.')
        return 1
    
    # Read per-frame scores from VMAF output and print.
    with open(logfile) as f:
        vmaf_data = json.load(f)
        vmaf_scores = []
        for frame in vmaf_data['frames']:
            vmaf_scores.append(frame['metrics']['vmaf'])
        print('RESULT VMAF: %s=' % vmaf_scores)
    
    return 0


def main():
    """The main function.
    
    A simple invocation is:
    ./webrtc/rtc_tools/compare_videos.py
    --ref_video=<path_and_name_of_reference_video>
    --test_video=<path_and_name_of_test_video>
    --frame_analyzer=<path_and_name_of_the_frame_analyzer_executable>
    Running vmaf requires the following arguments:
    --vmaf, --vmaf_model, --yuv_frame_width, --yuv_frame_height
    
    """
    
    try:
        # Directory to save temporary YUV files for VMAF in frame_analyzer.
        ## yuv_directory = tempfile.mkdtemp()
        yuv_directory  = '../VideosYUV/videoSRC001_640x360_30'
        _, vmaf_logfile = tempfile.mkstemp()
            
        # Run frame analyzer to compare the videos and print output.
        """if _RunFrameAnalyzer(options, yuv_directory=yuv_directory) != 0:
                #return 1
        """
            
        # Run VMAF for further video comparison and print output.
        _RunVmaf(yuv_directory, vmaf_logfile)
        
    finally:
        shutil.rmtree(yuv_directory)
        os.remove(vmaf_logfile)
            
    return 0

if __name__ == '__main__':
    sys.exit(main())