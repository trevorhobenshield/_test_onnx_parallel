function parseNpy(buffer) {
  const dtype = {
    '|u1': Uint8Array,
    '<i1': Int8Array,
    '|i1': Int8Array,
    '<u2': Uint16Array,
    '<i2': Int16Array,
    '<u4': Uint32Array,
    '<i4': Int32Array,
    '<f4': Float32Array,
    '<f8': Float64Array,
  };

  const magicLength = 6;
  const majorOffset = 6;
  const minorOffset = 7;
  const offsetLength = 8;

  const u8 = new Uint8Array(buffer);
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 6));
  if (magic !== '\x93NUMPY') {
    throw new Error(`Invalid .npy format: magic = ${magic}`);
  }

  const major = u8[majorOffset];
  const minor = u8[minorOffset];
  const lenLength = major >= 2 ? 4 : 2;
  const view = new DataView(buffer);

  let headerLen;
  if (major === 1 && minor === 0) {
    headerLen = view.getUint16(offsetLength, true);
  } else if ([2, 3].includes(major) && minor === 0) {
    headerLen = view.getUint32(offsetLength, true);
  } else {
    throw new Error(`Unsupported .npy version: ${major}.${minor}`);
  }

  const decoder = new TextDecoder(major === 3 && minor === 0 ? 'utf-8' : 'ascii');
  const headerText = decoder.decode(u8.subarray(offsetLength + lenLength, offsetLength + lenLength + headerLen));
  const headerJson = headerText
    .toLowerCase()
    .replace(/'/g, '"')
    .replace('(', '[')
    .replace(/,*\),*/g, ']');
  const {descr, shape, fortran_order} = JSON.parse(headerJson);

  if (fortran_order) {
    throw new Error('Fortran-ordered arrays not supported');
  }

  const start = magicLength + 2 + lenLength + headerLen;
  const data = new dtype[descr](buffer, start);

  return {
    major,
    minor,
    descr,
    dims: shape,
    data,
  };
}


/**
 * // Usage
 * const f32 = TensorType.FLOAT;
 * const shape = [2, 5];
 * const tensor = RandomTensor[f32](2, 5);
 * @type {{[p: string]: function(...[*]): *}}
 */
const RandomTensor = Object.fromEntries(Object.entries(TENSOR_TYPE_MAP).map(([dtype, arrayType]) => [dtype, (...shape) => {
  dtype = Number(dtype);
  const size = shape.reduce((a, b) => a * b, 1);
  const isFloat = dtype === TensorType.FLOAT || dtype === TensorType.DOUBLE;
  const isBool = dtype === TensorType.BOOL;
  let counter = 1;
  return new ort.Tensor(arrayType.from({length: size}, () => {
    if (isFloat) {
      return Math.random() * 2 - 1;
    } else if (isBool) {
      return Math.random() > 0.5 ? 1 : 0;
    } else {
      return counter++;
    }
  }), shape);
}]));

function getInOut(model) {
  function fn(k) {
    return Object.fromEntries(model.graph[k].map(({name, type}) => [name, {
      type: TENSOR_TYPE_INV[type.tensorType.elemType],
      dims: type.tensorType.shape.dim.map(y => y.dimValue),
    }]));
  }

  return [fn('input'), fn('output')];
}

function getInfo(file, model) {
  console.log(`%c${file.name}`, 'color: #db2777');
  console.log('      IR Version:', model.irVersion);
  console.log('   Opset Version:', model.opsetImport[0].version);
  console.log('        Producer:', model.producerName);
  console.log('Producer Version:', model.producerVersion);
  console.log('      Graph Name:', model.graph.name);
  [modelOps, unsupportedOps] = checkOps(model);
  console.log(`       Operators:`, modelOps);
  const [ins, outs] = getInOut(model);
  fmtInOut(ins, outs);
  console.log(`         Backend: ${SESSION_OPTIONS.executionProviders}`);
  console.log('model =', model);
  console.log('session =', session);
}

function mapInOut() {
  const M = {};
  for (let name in OpDef) {
    const op = OpDef[name][Object.keys(OpDef[name]).at(-1)];
    let inputs = [];
    let outputs = [];
    try {
      inputs = Object.keys(op.input);
    } catch (e) {
    }
    try {
      outputs = Object.keys(op.output);
    } catch (e) {
    }
    M[name] = {inputs: inputs, outputs: outputs};
  }
  return M;
}

function fmtInOut(ins, outs) {
  let formatParts = [];
  let colors = [];
  Object.values(ins).forEach(({type, dims}) => {
    formatParts.push(`%c[${TENSOR_TYPE_ABV[type]}]%c(${dims.map(x => x || '?').join(', ')})`);
    colors.push('color: #4d7c0f', null);
  });
  console.log(' '.repeat(14) + `In: ${formatParts.join(' ')}`, ...colors);
  formatParts = [];
  colors = [];
  Object.values(outs).forEach(({type, dims}) => {
    formatParts.push(`%c[${TENSOR_TYPE_ABV[type]}]%c(${dims.map(x => x || '?').join(', ')})`);
    colors.push('color: #4d7c0f', null);
  });
  console.log(' '.repeat(13) + `Out: ${formatParts.join(' ')}`, ...colors);
}

function checkOps(model) {
  const ops = [...new Set(model.graph.node.map(x => x.opType))].sort();
  const unsupported = ops.filter(x => !WEBGL_SUPPORTED_OPS.has(x)).sort();
  const backends = SESSION_OPTIONS.executionProviders;
  if (backends.includes('webgl') && unsupported.length) {
    console.error('Model contains unsupported operators', unsupported);
  }
  return [ops, unsupported];
}

function initOpLogger(session) {
  const inoutMap = mapInOut();
  const opLogHandler = {
    apply: async function(target, thisArg, args) {
      const [_handler, inputs, attributes] = args;
      const inputStrs = inputs.map(x => `(${x.dims})`).join(' ');
      const opType = thisArg.impl.__debug_name;
      const result = target.apply(thisArg, args);
      let outputStrs;
      let outputs;
      if (result instanceof Promise) {
        outputs = await result;
        outputStrs = outputs.map(x => `(${x.dims})`).join(' ');
      } else {
        outputs = result;
        outputStrs = outputs.map(x => `(${x.dims})`).join(' ');
      }
      console.groupCollapsed(`%c${inputStrs}%c`.padEnd(50, ' ') + ` => ${opType} => ` + ` %c${outputStrs}`.padStart(50 - opType.length, ' '), 'color:#f59e0b', '', 'color:#ea580c');
      console.groupCollapsed('attributes');
      console.log(attributes);
      console.groupEnd();
      console.groupCollapsed('inputs');
      try {
        inoutMap[opType].inputs.map((e, i) => {
          console.log(e, inputs[i]);
        });
      } catch (e) {
      }
      console.groupEnd();
      console.groupCollapsed('output');
      try {
        inoutMap[opType].outputs.map((e, i) => {
          console.log(e, outputs[i]);
        });
      } catch (e) {
      }
      console.groupEnd();
      console.groupEnd();
      return result;
    },
  };

  // trace Op implementation functions
  const opTypes = session.handler.session._model._graph._nodes.map(x => x.opType);
  for (let [name, op] of zip(opTypes, session.handler.session._ops)) {
    op.impl = new Proxy(op.impl, opLogHandler);
    op.impl.__debug_name = name; // attach debug attribute for logging
  }
}

function strToJSON(s) {
  const st = [];
  const r = [];
  let cA = r;
  let n = '';
  for (const c of s) {
    if (c === '[') {
      st.push(cA);
      cA = [];
    } else if (c === ']') {
      if (n) {
        cA.push(+n);
      }
      n = '';
      let tmp = cA;
      cA = st.pop();
      cA.push(tmp);
    } else if (c === ',' || c === ' ') {
      n && cA.push(+n);
      n = '';
    } else {
      n += c;
    }
  }
  n && cA.push(+n);
  return r;
}

function getShape(arr) {
  let res = [];
  let tmp = arr;
  while (Array.isArray(tmp)) {
    res.push(tmp.length);
    tmp = tmp[0];
  }
  return res;
}

function _zip(arr, ...arrs) {
  return arr.map((x, i) => [x, ...arrs.map(a => a[i])]);
}

function softmax(logits) {
  let max = Math.max(...logits);
  let scores = logits.map(x => Math.exp(x - max));
  let sum = scores.reduce((a, b) => a + b, 0);
  return scores.map(s => s / sum);
}

function getPredictions(output) {
  const res = [];
  const data = softmax(output);
  const preds = _zip(Array.from(data), in1k_classes).sort((a, b) => b[0] - a[0]);
  for (let p of preds.slice(0, 5)) {
    // console.log(p[0].toFixed(4), p[1]);
    res.push({label: p[1], prob: p[0]});
  }
  return res;
}