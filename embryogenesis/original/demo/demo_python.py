# WebGL Demo

# This code exports quantized models for the WebGL demo that is used in the article.
# The demo code can be found at https://github.com/distillpub/post--growing-ca/blob/master/public/ca.js

import base64
import json

import numpy as np


def pack_layer(weight, bias, outputType=np.uint8):
    in_ch, out_ch = weight.shape
    assert (in_ch % 4 == 0) and (out_ch % 4 == 0) and (bias.shape == (out_ch,))
    weight_scale, bias_scale = 1.0, 1.0
    if outputType == np.uint8:
        weight_scale = 2.0 * np.abs(weight).max()
        bias_scale = 2.0 * np.abs(bias).max()
        weight = np.round((weight / weight_scale + 0.5) * 255)
        bias = np.round((bias / bias_scale + 0.5) * 255)
    packed = np.vstack([weight, bias[None, ...]])
    packed = packed.reshape(in_ch + 1, out_ch // 4, 4)
    packed = outputType(packed)
    packed_b64 = base64.b64encode(packed.tobytes()).decode('ascii')
    return {'data_b64': packed_b64, 'in_ch': in_ch, 'out_ch': out_ch,
            'weight_scale': weight_scale, 'bias_scale': bias_scale,
            'type': outputType.__name__}


def export_ca_to_webgl_demo(ca, outputType=np.uint8):
    # reorder the first layer inputs to meet webgl demo perception layout
    chn = ca.channel_n
    w1 = ca.weights[0][0, 0].numpy()
    w1 = w1.reshape(chn, 3, -1).transpose(1, 0, 2).reshape(3 * chn, -1)
    layers = [
        pack_layer(w1, ca.weights[1].numpy(), outputType),
        pack_layer(ca.weights[2][0, 0].numpy(), ca.weights[3].numpy(), outputType)
    ]
    return json.dumps(layers)


with zipfile.ZipFile('webgl_models8.zip', 'w') as zf:
    for e in EMOJI:
        zf.writestr('ex1_%s.json' % e, export_ca_to_webgl_demo(get_model(e, use_pool=0, damage_n=0)))
        run = 1 if e in '😀🕸' else 0  # select runs that happen to quantize better
        zf.writestr('ex2_%s.json' % e, export_ca_to_webgl_demo(get_model(e, use_pool=1, damage_n=0, run=run)))
        run = 1 if e in '🦎' else 0  # select runs that happen to quantize better
        zf.writestr('ex3_%s.json' % e, export_ca_to_webgl_demo(get_model(e, use_pool=1, damage_n=3, run=run)))
