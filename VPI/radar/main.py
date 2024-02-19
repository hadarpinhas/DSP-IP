# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
import vpi
import numpy as np
from PIL import Image
from argparse import ArgumentParser

# ----------------------------
# Parse command line arguments

parser = ArgumentParser()
parser.add_argument('backend', choices=['cpu','cuda'],
                    help='Backend to be used for processing')

parser.add_argument('input',
                    help='Input image in space domain')

args = parser.parse_args();

if args.backend == 'cpu':
    backend = vpi.Backend.CPU
else:
    assert args.backend == 'cuda'
    backend = vpi.Backend.CUDA

# --------------------------------------------------------------
# Load input into a vpi.Image and convert it to float grayscale
with vpi.Backend.CUDA:
    try:
        input = vpi.asimage(np.asarray(Image.open(args.input))).convert(vpi.Format.F32)
    except IOError:
        sys.exit("Input file not found")
    except:
        sys.exit("Error with input file")

# --------------------------------------------------------------
# Transform input into frequency domain
with backend:
    hfreq = input.fft()

# --------------------------------------------------------------
# Post-process results and save to disk

# Transform [H,W,2] float array into [H,W] complex array
hfreq = hfreq.cpu().view(dtype=np.complex64).squeeze(2)

# Complete array into a full hermitian matrix
if input.width%2==0:
    wpad = input.width//2-1
    padmode = 'reflect'
else:
    wpad = input.width//2
    padmode='symmetric'
freq = np.pad(hfreq, ((0,0),(0,wpad)), mode=padmode)
freq[:,hfreq.shape[1]:] = np.conj(freq[:,hfreq.shape[1]:])
freq[1:,hfreq.shape[1]:] = freq[1:,hfreq.shape[1]:][::-1]

# Shift 0Hz to image center
freq = np.fft.fftshift(freq)

# Convert complex frequencies into log-magnitude
lmag = np.log(1+np.absolute(freq))

# Normalize into [0,255] range
min = lmag.min()
max = lmag.max()
lmag = ((lmag-min)*255/(max-min)).round().astype(np.uint8)

# -------------------
# Save result to disk
Image.fromarray(lmag).save('spectrum_python'+str(sys.version_info[0])+'_'+args.backend+'.png')
