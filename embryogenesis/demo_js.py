import glob

# TensorFlow.js Demo {run:"auto", vertical-output: true}
# Select "CHECKPOINT" model to load the checkpoint created by running cells from the "Training" section of this notebook
import IPython.display

# Available models checkpoint: ['CHECKPOINT', '😀 1F600', '💥 1F4A5', '👁 1F441', '🦎 1F98E', '🐠 1F420', '🦋 1F98B',
# '🐞 1F41E', '🕸 1F578', '🥨 1F968', '🎄 1F384']
model = "CHECKPOINT"
model_type = '3 regenerating'  # ['1 naive', '2 persistent', '3 regenerating']

# Shift-click to seed the pattern

if model != 'CHECKPOINT':
    code = model.split(' ')[1]
    emoji = chr(int(code, 16))
    experiment_i = int(model_type.split()[0]) - 1
    use_pool = (0, 1, 1)[experiment_i]
    damage_n = (0, 0, 3)[experiment_i]
    model_str = get_model(emoji, use_pool=use_pool, damage_n=damage_n, output='json')
else:
    last_checkpoint_fn = sorted(glob.glob('train_log/*.json'))[-1]
    model_str = open(last_checkpoint_fn).read()

data_js = '''
  window.GRAPH_URL = URL.createObjectURL(new Blob([`%s`], {type: 'application/json'}));
''' % (model_str)

display(IPython.display.Javascript(data_js))

IPython.display.HTML('''
<script src="https://unpkg.com/@tensorflow/tfjs-core@latest/dist/tf-core.js"></script>
<script src="https://unpkg.com/@tensorflow/tfjs-layers@latest/dist/tf-layers.js"></script>
<script src="https://unpkg.com/@tensorflow/tfjs-converter@latest/dist/tf-converter.js"></script>
<script src="https://unpkg.com/@tensorflow/tfjs-backend-wasm@latest/dist/tf-backend-wasm.js"></script>

<canvas id='canvas' style="border: 1px solid black; image-rendering: pixelated;"></canvas>

<script>
  "use strict";

  const sleep = (ms)=>new Promise(resolve => setTimeout(resolve, ms));

  const parseConsts = model_graph=>{
    const dtypes = {'DT_INT32':['int32', 'intVal', Int32Array],
                    'DT_FLOAT':['float32', 'floatVal', Float32Array]};

    const consts = {};
    model_graph.modelTopology.node.filter(n=>n.op=='Const').forEach((node=>{
      const v = node.attr.value.tensor;
      const [dtype, field, arrayType] = dtypes[v.dtype];
      if (!v.tensorShape.dim) {
        consts[node.name] = [tf.scalar(v[field][0], dtype)];
      } else {
        const shape = v.tensorShape.dim.map(d=>parseInt(d.size));
        let arr;
        if (v.tensorContent) {
          const data = atob(v.tensorContent);
          const buf = new Uint8Array(data.length);
          for (var i=0; i<data.length; ++i) {
            buf[i] = data.charCodeAt(i);
          }
          arr = new arrayType(buf.buffer);
        } else {
          const size = shape.reduce((a, b)=>a*b);
          arr = new arrayType(size);
          arr.fill(v[field][0]);
        }
        consts[node.name] = [tf.tensor(arr, shape, dtype)];
      }
    }));
    return consts;
  }

  const run = async ()=>{
    const r = await fetch(GRAPH_URL);
    const consts = parseConsts(await r.json());

    const model = await tf.loadGraphModel(GRAPH_URL);
    Object.assign(model.weights, consts);

    let seed = new Array(16).fill(0).map((x, i)=>i<3?0:1);
    seed = tf.tensor(seed, [1, 1, 1, 16]);

    const D = 96;
    const initState = tf.tidy(()=>{
      const D2 = D/2;
      const a = seed.pad([[0, 0], [D2-1, D2], [D2-1, D2], [0,0]]);
      return a;
    });

    const state = tf.variable(initState);
    const [_, h, w, ch] = state.shape;

    const damage = (x, y, r)=>{
      tf.tidy(()=>{
        const rx = tf.range(0, w).sub(x).div(r).square().expandDims(0);
        const ry = tf.range(0, h).sub(y).div(r).square().expandDims(1);
        const mask = rx.add(ry).greater(1.0).expandDims(2);
        state.assign(state.mul(mask));
      });
    }

    const plantSeed = (x, y)=>{
      const x2 = w-x-seed.shape[2];
      const y2 = h-y-seed.shape[1];
      if (x<0 || x2<0 || y2<0 || y2<0)
        return;
      tf.tidy(()=>{
        const a = seed.pad([[0, 0], [y, y2], [x, x2], [0,0]]);
        state.assign(state.add(a));
      });
    }

    const scale = 4;

    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = `${w*scale}px`;
    canvas.style.height = `${h*scale}px`;

    canvas.onmousedown = e=>{
      const x = Math.floor(e.clientX/scale);
        const y = Math.floor(e.clientY/scale);
        if (e.buttons == 1) {
          if (e.shiftKey) {
            plantSeed(x, y);  
          } else {
            damage(x, y, 8);
          }
        }
    }
    canvas.onmousemove = e=>{
      const x = Math.floor(e.clientX/scale);
      const y = Math.floor(e.clientY/scale);
      if (e.buttons == 1 && !e.shiftKey) {
        damage(x, y, 8);
      }
    }

    for (let i=0;; ++i) {
      if (i%2==0) {
        const imageData = tf.tidy(()=>{
          const rgba = state.slice([0, 0, 0, 0], [-1, -1, -1, 4]);
          const a = state.slice([0, 0, 0, 3], [-1, -1, -1, 1]);
          const img = tf.tensor(1.0).sub(a).add(rgba).mul(255);
          const rgbaBytes = new Uint8ClampedArray(img.dataSync());
          return new ImageData(rgbaBytes, w, h);
        });
        ctx.putImageData(imageData, 0, 0);
        await sleep(1);
      }
      tf.tidy(()=>{
        state.assign(model.execute(
            {x:state, fire_rate:tf.tensor(0.5),
            angle:tf.tensor(0.0), step_size:tf.tensor(1.0)}, ['Identity']));
      });
    }
  }
  //tf.setBackend('wasm').then(run);
  run();

</script>
''')
