function changeExpression(mouth, eyes) {
  return function(imageClass) {
    switch (imageClass) {
      case 0:
        mouth.className = 'mouth neutral-mouth';
        eyes.forEach((eye) => eye.className = 'eye neutral-eyes');
        break;
      case 1:
        mouth.className = 'mouth smile-mouth';
        eyes.forEach((eye) => eye.className = 'eye smile-eyes');
        break;
      case 2:
        mouth.className = 'mouth frown-mouth';
        eyes.forEach((eye) => eye.className = 'eye frown-eyes');
        break;
    }
  }
}

// Takes an ImageData array, which contains four elements for each pixel (R, G, B, and A values)
// Outputs an array one fourth the size, with the single greyscaled value for each pixel
function grayscale(pixels) {
  const grayPixels = [];
  const length     = pixels.length
  let i;

  for (i = 0; i < length; i += 4) {
    grayPixels.push((pixels[i] * 0.299) + (pixels[i + 1] * 0.587) + (pixels[i + 2] * 0.114));
  }

  return grayPixels;
};

function classify(grayscalePixels) {
  return  Math.floor(Math.random() * 3);
}

window.addEventListener('load', function(event) {
  const mouth   = document.querySelector('.mouth');
  const eyes    = document.querySelectorAll('.eye');
  const animate = changeExpression(mouth, eyes);
  const video   = document.querySelector('video');
  const canvas  = document.querySelector('canvas');
  const context = canvas.getContext('2d');
  context.scale(0.125, 0.125); // we want to feed a 48x48 image to the classifier

  navigator.mediaDevices
           .getUserMedia({ video: { width: 384, height: 384 } })
           .then(function(mediaStream) {
             video.srcObject = mediaStream;
             video.play();

             setInterval(function() {
               context.drawImage(video, 0, 0, 384, 384);
               const pixels          = context.getImageData(0, 0, 48, 48).data;
               const grayscalePixels = grayscale(pixels);
               const imageClass      = classify(grayscalePixels);

               animate(imageClass);
             }, 1000);
           })
           .catch(function(err) {
             console.error('Error getting video stream:', err.message);
           });
});
