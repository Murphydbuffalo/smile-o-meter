function changeExpression(mouth, eyes) {
  return function(event) {
    if (mouth.className.includes('frown-mouth')) {
      mouth.className = 'mouth neutral-mouth';
      eyes.forEach((eye) => eye.className = 'eye neutral-eyes');
    } else if (mouth.className.includes('neutral-mouth')) {
      mouth.className = 'mouth smile-mouth';
      eyes.forEach((eye) => eye.className = 'eye smile-eyes');
    } else {
      mouth.className = 'mouth frown-mouth';
      eyes.forEach((eye) => eye.className = 'eye frown-eyes');
    }
  }
}

window.addEventListener("load", function(event) {
  const mouth = document.querySelector('.mouth');
  const eyes  = document.querySelectorAll('.eye');

  document.querySelector('.smiley').onclick = changeExpression(mouth, eyes)
  navigator.mediaDevices
           .getUserMedia({ video: true })
           .then(function(mediaStream) {
             const video     = document.querySelector('video');
             video.srcObject = mediaStream;
             video.play();
           })
           .catch(function(err) {
             console.error('Error getting video stream:', err.message);
           });
});
