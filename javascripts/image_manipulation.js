const ImageManipulation = class ImageManipulation {
  constructor(videoStream, videoDimension) {
    this.videoStream                 = videoStream;
    this.videoDimension              = videoDimension;
    this.neuralNetworkImageDimension = 48;
  }

  renderGrayscale(grayscalePixels) {
    const rgbaPixels = this.rgbaFromGrayscale(grayscalePixels);
    const imageData  = this.canvasContext().createImageData(this.neuralNetworkImageDimension, this.neuralNetworkImageDimension);
    imageData.data.set(rgbaPixels);
    this.canvasContext().putImageData(imageData, 0, 0);
  }

  // Takes an ImageData array, which contains four elements for each pixel (R, G, B, and A values)
  // Outputs an array one fourth the size, with the single greyscaled value for each pixel
  grayscale(pixels) {
    const grayPixels = [];
    const length     = pixels.length
    let i;

    for (i = 0; i < length; i += 4) {
      grayPixels.push((pixels[i] * 0.299) + (pixels[i + 1] * 0.587) + (pixels[i + 2] * 0.114));
    }

    return grayPixels;
  };

  rgbaFromGrayscale(pixels) {
    const buffer = new Uint8ClampedArray(pixels.length * 4);
    const length = pixels.length
    let grayscaleIndex, bufferIndex;

    for (grayscaleIndex = 0; grayscaleIndex < length; grayscaleIndex += 1) {
      bufferIndex = grayscaleIndex * 4;
      pixel       = pixels[grayscaleIndex];
      buffer[bufferIndex]     = pixel // Red
      buffer[bufferIndex + 1] = pixel // Green
      buffer[bufferIndex + 2] = pixel // Blue
      buffer[bufferIndex + 3] = 255;  // Alpha
    }

    return buffer;
  }

  start() {
    this.video.srcObject = this.videoStream;
    this.video.play();
  }

  get currentFrame() {
    this.canvasContext().drawImage(this.video, 0, 0, this.videoDimension, this.videoDimension);
    return this.canvasContext().getImageData(0, 0, this.neuralNetworkImageDimension, this.neuralNetworkImageDimension).data;
  }

  get video() {
    return document.querySelector('video');
  }

  get canvas() {
    return document.querySelector('canvas');
  }

  canvasContext() {
    if (this.context != null) {
      return this.context;
    }

    const context = this.canvas.getContext('2d');
    context.scale(this.scalingFactor, this.scalingFactor);

    if (window.location.search.includes('show-grayscale')) {
      this.canvas.className = 'show';
    }

    this.context = context;

    return context;
  }

  get scalingFactor() {
    // we need to scale the images from the video feed to match the dimensions expected by the neural network
    return this.neuralNetworkImageDimension / this.videoDimension;
  }
}

export default ImageManipulation;
