importScripts(
  'https://cdn.jsdelivr.net/npm/jimp@0.22.10/browser/lib/jimp.min.js',
);


self.onmessage = async (e) => {
  if (e.data.msg === 'processImage') {
    const {file, inputDims} = e.data;
    const blobUrl = URL.createObjectURL(file);
    const jimpImage = await Jimp.read(blobUrl);
    const resizedImage = jimpImage.resize(...inputDims);
    const buf = await resizedImage.getBufferAsync(Jimp.MIME_JPEG);
    const bitmap = await createImageBitmap(new Blob([buf]));
    URL.revokeObjectURL(blobUrl); // free?
    self.postMessage({
      msg: 'imageProcessed',
      file,
      bitmap,
    }, [bitmap]);
  }
};

