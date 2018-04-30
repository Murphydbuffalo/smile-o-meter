import ImageManipulation from './image_manipulation.js';


const animate = function(mouth, eyes, imageClass) {
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
};

const classify = function(grayscalePixels) {
  // TODO: Call AWS Lambda with neural net code
  return  Math.floor(Math.random() * 3);
};

window.addEventListener('load', function(event) {
  const mouth          = document.querySelector('.mouth');
  const eyes           = document.querySelectorAll('.eye');
  const videoDimension = 384.0;

  navigator.mediaDevices
           .getUserMedia({ video: { width: videoDimension, height: videoDimension } })
           .then(function(mediaStream) {
             const videoFeed = new ImageManipulation(mediaStream, videoDimension);
             videoFeed.start();

             setInterval(function() {
               const pixels          = videoFeed.currentFrame;
               const grayscalePixels = videoFeed.grayscale(pixels);
               const imageClass      = classify(grayscalePixels);

               if (window.location.search.includes('show-grayscale')) {
                 videoFeed.renderGrayscale(grayscalePixels);
               }

               animate(mouth, eyes, imageClass);
             }, 1000);
           })
           .catch(function(err) {
             console.error('Error getting video stream:', err.message);
           });
});
