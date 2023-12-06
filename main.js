let model, inputDims, session, counter = 1, numFiles = 0, predictions = [];
let workers = [], N_WORKERS = navigator.hardwareConcurrency || 1;
let t0;

canvas = document.createElement('canvas');
ctx = canvas.getContext('2d');
pbar = document.querySelector('#progress-bar');
downloadButton = document.querySelector('#download');
uploadButton = document.querySelector('#upload');
modelSelect = document.querySelector('#modelSelect');

// dynamically populate model dropdown menu
MODEL_PATHS.forEach(model => {
    const option = document.createElement('option');
    option.value = `${MODEL_ROOT_PATH}/${model}`;
    option.textContent = model;
    modelSelect.appendChild(option);
});


// load default model
(async () => {
    const modelName = document.querySelector('#modelSelect').value;
    const r = await fetch(modelName);
    buffer = await r.arrayBuffer();

    model = ModelProto.decode(new Uint8Array(buffer));
    inputDims = model.graph.input[0].type.tensorType.shape.dim.map(x => x.dimValue).slice(2, 4);
    ModelProto.verify(model);
    console.log('model =', model);

    session = await ort.InferenceSession.create(buffer, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
        executionMode: 'parallel',
        enableCpuMemArena: true,
        enableMemPattern: true,
        extra: {
            session: {
                set_denormal_as_zero: '1',
                disable_prepacking: '1',
            },
            optimization: {
                enable_gelu_approximation: '1',
            },
        },
    });
})();


document.querySelector('#modelSelect').onchange = async () => {
    const modelName = document.querySelector('#modelSelect').value;
    if (!modelName) return; // Do nothing if no model is selected
    try {
        const r = await fetch(modelName);
        const buffer = await r.arrayBuffer();

        model = ModelProto.decode(new Uint8Array(buffer));
        inputDims = model.graph.input[0].type.tensorType.shape.dim.map(x => x.dimValue).slice(2, 4);
        ModelProto.verify(model);
        console.log('model =', model);
        resetProgress();

        session = await ort.InferenceSession.create(buffer, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
            executionMode: 'parallel',
            enableCpuMemArena: true,
            enableMemPattern: true,
            extra: {
                session: {
                    set_denormal_as_zero: '1',
                    disable_prepacking: '1',
                },
                optimization: {
                    enable_gelu_approximation: '1',
                },
            },
        });
        console.log('session =', session);
    } catch (e) {
        console.error('Error loading model:', e);
    }
};

async function workerHandler(e) {
    if (e.data.msg === 'imageProcessed') {
        const {file, bitmap} = e.data;
        const out = await session.run({
            [session.inputNames[0]]: await ort.Tensor.fromImage(bitmap, {
                tensorFormat: 'RGB',
                tensorLayout: 'NCHW',
                dataType: 'float32',
            }),
        });
        updateProgress();
        const preds = getPredictions(out[session.outputNames[0]].data);
        console.log('preds =', preds);
        predictions.push({file, out: preds});
        if (numFiles === counter) {
            console.log(performance.now() - t0);
        }
    }
}

uploadButton.onchange = async (e) => {
    t0 = performance.now();
    resetProgress();
    const files = [...e.target.files];
    numFiles = files.length;
    for (let i = 0; i < N_WORKERS; i++) {
        try {
            const worker = new Worker('worker.js');
            worker.onmessage = workerHandler;
            workers.push(worker);
        } catch (err) {
            console.log('Error creating worker:', err);
        }
    }

    for (let i = 0; i < files.length; i++) {
        const idx = i % N_WORKERS;
        workers[idx].postMessage({msg: 'processImage', file: files[i], inputDims});
    }
    e.target.value = null; // reset input
};


downloadButton.onclick = async () => {
    const writer = new zip.ZipWriter(new zip.BlobWriter('application/zip'));
    const promises = predictions.map(async ({file, out}, i) => {
        const cls = out[0].label;
        await writer.add(`${cls}/${file.name}`, new zip.BlobReader(new Blob([file])));
    });
    await Promise.all(promises);
    const blob = await writer.close();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'dataset.zip';
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
};

function updateProgress() {
    pbar.style.width = (counter / numFiles) * 100 + '%';
    pbar.style.backgroundColor = '#4CAF50';
    counter++;
}

function resetProgress() {
    counter = 0;
    updateProgress();
}
