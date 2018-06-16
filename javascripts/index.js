import ImageManipulation from './image_manipulation.js';

const animate = function(mouth, eyes, facialExpression) {
  mouth.className = `mouth ${facialExpression}-mouth`;
  eyes.forEach((eye) => eye.className = `eye ${facialExpression}-eyes`);
};

// FER labels:
// 0 = Angry, 1 = Disgusted, 2 = Afraid, 3 = Happy, 4 = Sad, 5 = Surprised, 6 = Neutral
const labelToFacialExpression = function(ferLabel) {
  console.log('class returned by classifier is', ferLabel);
  switch (ferLabel) {
    case 6:
      return 'neutral';
    case 3:
      return 'smile';
    else
      return 'frown';
  }
};

const classify = function(grayscalePixels) {
  const url     = "https://295xgmxf2m.execute-api.us-east-1.amazonaws.com/V1/predict-dev-predict";
  const options = {
    body:    JSON.stringify({ pixels: grayscalePixels }),
    cache:   'no-cache',
    headers: { 'content-type': 'application/json', 'accept': 'application/json' },
    method:  'POST',
    mode:    'cors'
  };

  return fetch(url, options).then(response   => response.json())
                            .then(imageClass => labelToFacialExpression(imageClass))
                            .catch(error     => console.error('Error calling lambda:', error.message));
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

               classify(grayscalePixels).then(function(emotion) {
                 console.log('emotion is', emotion);

                 if (window.location.search.includes('show-grayscale')) {
                   videoFeed.renderGrayscale(grayscalePixels);
                 }

                 animate(mouth, eyes, emotion);
               });
             }, 1000);
           })
           .catch(function(err) {
             console.error('Error getting video stream:', err.message);
           });
});
